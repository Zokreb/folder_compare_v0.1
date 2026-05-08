import os
import sqlite3
import shutil
from pathlib import Path
import cv2
import numpy as np
from PIL import Image
import imagehash

class ImageProcessor:
    def __init__(self, db_path="temp_image_cache.db"):
        self.db_path = db_path
        self.conn = sqlite3.connect(self.db_path, check_same_thread=False)
        self.init_db()

    def init_db(self):
        cursor = self.conn.cursor()
        # Table for storing scanned image metadata
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS images (
                id INTEGER PRIMARY KEY,
                folder_type TEXT,
                filepath TEXT,
                filename TEXT,
                size_bytes INTEGER,
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

    def scan_folder(self, folder_path, folder_type, progress_callback=None):
        folder = Path(folder_path)
        valid_exts = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
        files = [f for f in folder.iterdir() if f.suffix.lower() in valid_exts]
        
        total = len(files)
        cursor = self.conn.cursor()
        
        for i, filepath in enumerate(files):
            try:
                with Image.open(filepath) as img:
                    width, height = img.size
                    h = str(imagehash.phash(img, hash_size=8))
                
                size_bytes = filepath.stat().st_size
                
                cursor.execute('''
                    INSERT INTO images (folder_type, filepath, filename, size_bytes, width, height, phash)
                    VALUES (?, ?, ?, ?, ?, ?, ?)
                ''', (folder_type, str(filepath), filepath.name, size_bytes, width, height, h))
                
            except Exception as e:
                print(f"Error reading {filepath}: {e}")
            
            if progress_callback:
                progress_callback(int((i + 1) / total * 100))
                
        self.conn.commit()

    def find_matches(self, threshold=10):
        cursor = self.conn.cursor()
        cursor.execute("SELECT id, phash FROM images WHERE folder_type='original'")
        originals = cursor.fetchall()
        
        cursor.execute("SELECT id, phash FROM images WHERE folder_type='cropped'")
        cropped = cursor.fetchall()

        for c_id, c_hash_str in cropped:
            c_hash = imagehash.hex_to_hash(c_hash_str)
            best_match = None
            min_dist = threshold

            for o_id, o_hash_str in originals:
                o_hash = imagehash.hex_to_hash(o_hash_str)
                dist = c_hash - o_hash
                if dist < min_dist:
                    min_dist = dist
                    best_match = o_id

            if best_match is not None:
                cursor.execute("INSERT INTO matches (orig_id, crop_id, distance) VALUES (?, ?, ?)", 
                               (best_match, c_id, min_dist))
        self.conn.commit()

    def get_match_pairs(self):
        cursor = self.conn.cursor()
        cursor.execute('''
            SELECT 
                o.filepath, o.filename, o.size_bytes, o.width, o.height,
                c.filepath, c.filename, c.size_bytes, c.width, c.height
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
        
        # Move cropped to trash
        shutil.move(str(crop_p), str(trash_dir / crop_p.name))
        
        # Copy original to cropped folder, using original's extension
        new_name = crop_p.stem + orig_p.suffix
        shutil.copy2(str(orig_p), str(crop_p.parent / new_name))

    def generate_overlay(self, orig_path, crop_path):
        """Uses ORB to find the cropped area and draw a box on the original image"""
        img_orig = cv2.imread(orig_path)
        img_crop = cv2.imread(crop_path)
        
        if img_orig is None or img_crop is None:
            return None

        # Convert to grayscale
        gray_orig = cv2.cvtColor(img_orig, cv2.COLOR_BGR2GRAY)
        gray_crop = cv2.cvtColor(img_crop, cv2.COLOR_BGR2GRAY)

        orb = cv2.ORB_create()
        kp1, des1 = orb.detectAndCompute(gray_crop, None)
        kp2, des2 = orb.detectAndCompute(gray_orig, None)

        bf = cv2.BFMatcher(cv2.NORM_HAMMING, crossCheck=True)
        try:
            matches = bf.match(des1, des2)
            matches = sorted(matches, key=lambda x: x.distance)
            
            # Extract locations of good matches
            src_pts = np.float32([kp1[m.queryIdx].pt for m in matches]).reshape(-1, 1, 2)
            dst_pts = np.float32([kp2[m.trainIdx].pt for m in matches]).reshape(-1, 1, 2)

            M, mask = cv2.findHomography(src_pts, dst_pts, cv2.RANSAC, 5.0)
            h, w = gray_crop.shape
            pts = np.float32([[0, 0], [0, h-1], [w-1, h-1], [w-1, 0]]).reshape(-1, 1, 2)
            
            if M is not None:
                dst = cv2.perspectiveTransform(pts, M)
                # Draw green bounding box on the original image
                img_with_box = cv2.polylines(img_orig, [np.int32(dst)], True, (0, 255, 0), 3, cv2.LINE_AA)
                return cv2.cvtColor(img_with_box, cv2.COLOR_BGR2RGB)
        except Exception:
            pass
            
        return cv2.cvtColor(img_orig, cv2.COLOR_BGR2RGB)