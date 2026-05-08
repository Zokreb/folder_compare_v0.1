import os
import sqlite3
import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import imagehash
import math

class ImageProcessor:
    def __init__(self, db_path="temp_image_cache.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.init_db()

    def init_db(self):
        cursor = self.conn.cursor()
        
        # Check if we are dealing with an old schema
        cursor.execute("PRAGMA table_info(images)")
        cols = [col[1] for col in cursor.fetchall()]
        if cols and 'mtime' not in cols:
            print("Old database schema detected. Dropping old tables...")
            cursor.execute("DROP TABLE IF EXISTS images")
            cursor.execute("DROP TABLE IF EXISTS matches")

        # Table for storing scanned image metadata (Now with UNIQUE filepath and mtime)
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY,
                folder_type TEXT,
                filepath TEXT UNIQUE,
                filename TEXT,
                size_bytes INTEGER,
                mtime REAL,
                width INTEGER,
                height INTEGER,
                phash TEXT
            )
        ''')
        # Table for storing matches
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS matches (
                orig_id INTEGER,
                crop_id INTEGER,
                distance INTEGER
            )
        ''')
        self.conn.commit()


    def get_existing_hash_size(self):
        """Mathematically derives the hash_size from the length of the stored hex strings."""
        cursor = self.conn.cursor()
        cursor.execute("SELECT phash FROM images LIMIT 1")
        row = cursor.fetchone()
        if row and row[0]:
            # Formula: length of hex string * 4 gives total bits. Square root gives the grid size.
            return int(math.sqrt(len(row[0]) * 4))
        return None  # Database is empty

    # UPDATE 1: Add hash_size=8 to the parameters
    def scan_folder(self, folder_path, folder_type, force_rebuild=False, hash_size=8, progress_callback=None):
        folder = Path(folder_path)
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        files = [f for f in folder.iterdir() if f.suffix.lower() in valid_exts]
        
        total = len(files)
        cursor = self.conn.cursor()

        if force_rebuild:
            cursor.execute("DELETE FROM images WHERE folder_type=?", (folder_type,))
            existing_cache = {}
        else:
            cursor.execute("SELECT filepath, size_bytes, mtime FROM images WHERE folder_type=?", (folder_type,))
            existing_cache = {row[0]: {'size': row[1], 'mtime': row[2]} for row in cursor.fetchall()}
        
        for i, filepath in enumerate(files):
            str_path = str(filepath)
            stat = filepath.stat()
            
            if str_path in existing_cache:
                cached = existing_cache[str_path]
                if cached['size'] == stat.st_size and cached['mtime'] == stat.st_mtime:
                    if progress_callback:
                        progress_callback(int((i + 1) / total * 100))
                    continue

            try:
                with Image.open(filepath) as img:
                    width, height = img.size
                    # UPDATE 2: Pass the dynamic hash_size variable here
                    h = str(imagehash.phash(img, hash_size=hash_size)) 
                
                cursor.execute('''
                    INSERT OR REPLACE INTO images 
                    (folder_type, filepath, filename, size_bytes, mtime, width, height, phash)
                    VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                ''', (folder_type, str_path, filepath.name, stat.st_size, stat.st_mtime, width, height, h))
                
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
            
            if progress_callback:
                progress_callback(int((i + 1) / total * 100))
                
        self.conn.commit()

    def clear_matches(self):
        cursor = self.conn.cursor()
        cursor.execute("DELETE FROM matches")
        self.conn.commit()

    def find_matches(self, threshold=10, progress_callback=None):
        self.clear_matches()
        cursor = self.conn.cursor()
        
        cursor.execute("SELECT id, phash FROM images WHERE folder_type='original'")
        originals = cursor.fetchall()
        
        cursor.execute("SELECT id, phash FROM images WHERE folder_type='cropped'")
        cropped = cursor.fetchall()

        total = len(cropped)
        if total == 0:
            if progress_callback:
                progress_callback(100)
            return

        for i, (c_id, c_hash_str) in enumerate(cropped):
            c_hash = imagehash.hex_to_hash(c_hash_str)
            best_match = None
            min_dist = threshold

            """for o_id, o_hash_str in originals:
                o_hash = imagehash.hex_to_hash(o_hash_str)
                dist = c_hash - o_hash
                if dist < min_dist:
                    min_dist = dist
                    best_match = o_id"""
            
            for o_id, o_hash_str in originals:
                o_hash = imagehash.hex_to_hash(o_hash_str)
                dist = c_hash - o_hash
                if dist < min_dist:
                    # FIX: Cast numpy.int64 to standard Python int
                    min_dist = int(dist) 
                    best_match = o_id

            if best_match is not None:
                cursor.execute("INSERT INTO matches (orig_id, crop_id, distance) VALUES (?, ?, ?)", 
                               (best_match, c_id, min_dist))
            
            # Emit progress after checking each cropped image against all originals
            if progress_callback:
                progress_callback(int((i + 1) / total * 100))
                
        self.conn.commit()

    def get_match_pairs(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT 
                o.filepath, o.filename, o.size_bytes, o.width, o.height,
                c.filepath, c.filename, c.size_bytes, c.width, c.height,
                m.distance
            FROM matches m
            JOIN images o ON m.orig_id = o.id
            JOIN images c ON m.crop_id = c.id
        ''')
        return cursor.fetchall()

    def replace_image(self, orig_path, crop_path):
        orig_p = Path(orig_path)
        crop_p = Path(crop_path)
        
        trash_dir = crop_p.parent / "_Trash"
        trash_dir.mkdir(exist_ok=True)
        
        shutil.move(str(crop_p), str(trash_dir / crop_p.name))
        
        new_name = crop_p.stem + orig_p.suffix
        shutil.copy2(str(orig_p), str(crop_p.parent / new_name))

    def generate_overlay(self, orig_path, crop_path):
        img_orig = cv2.imread(orig_path)
        img_crop = cv2.imread(crop_path)
        
        if img_orig is None or img_crop is None:
            return None

        gray_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
        gray_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(gray_crop, None)
        kp2, des2 = orb.detectAndCompute(gray_orig, None)

        if des1 is None or des2 is None:
            return cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        try:
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            h, w = gray_crop.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            
            if M is not None:
                dst = cv2.perspectiveTransform(pts, M)
                img_with_box = cv2.polylines(img_orig, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
                return cv2.cvtColor(img_with_box, cv2.COLOR_BGR2RGB)
        except Exception:
            pass
            
        return cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)