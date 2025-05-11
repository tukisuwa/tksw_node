import os
import random
import torch
import hashlib
import codecs

class TextFileSelector:
    def __init__(self):
        self.round_robin_index = 0
        self.last_folder_path = ""
        self.cached_file_list = []
        self.file_content_cache = {}
        self.cache_progress_index = 0
        self.last_list_hash = None

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "folder_path": ("STRING", {"multiline": False, "default": ""}),
                "mode": (["random", "round-robin"], {"default": "random"}),
                "seed": ("INT", {"default": 0, "min": 0, "max": 0xffffffffffffffff}),
                "reset_state": ("BOOLEAN", {"default": False}),
                "cache_chunk_size": ("INT", {"default": 10, "min": 0, "max": 1000}),
                "encoding": ("STRING", {"multiline": False, "default": "utf-8"}),
                "filename_filter": ("STRING", {"multiline": False, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING", "STRING")
    RETURN_NAMES = ("text", "filename")
    FUNCTION = "select_and_read_file"
    CATEGORY = "tksw_node"

    def _scan_folder(self, folder_path):
        txt_files = []
        try:
            print(f"[TextFileSelector] Scanning folder: {folder_path}")
            if not os.path.isdir(folder_path):
                 print(f"[TextFileSelector] Warning: Folder does not exist or is not a directory: {folder_path}")
                 return []
            for filename in os.listdir(folder_path):
                full_path = os.path.join(folder_path, filename)
                if os.path.isfile(full_path) and filename.lower().endswith(".txt"):
                    txt_files.append(filename)
            print(f"[TextFileSelector] Found {len(txt_files)} .txt files.")
            return sorted(txt_files)
        except Exception as e:
            print(f"[TextFileSelector] Error scanning folder '{folder_path}': {e}")
            return []

    def _read_and_cache_file(self, filename, folder_path, encoding):
        full_path = os.path.join(folder_path, filename)
        if filename in self.file_content_cache:
            return self.file_content_cache[filename]
        print(f"[TextFileSelector] Reading and caching: {filename} (Encoding: {encoding})")
        try:
            with codecs.open(full_path, 'r', encoding=encoding, errors='ignore') as f:
                content = f.read()
            self.file_content_cache[filename] = content
            return content
        except FileNotFoundError:
             print(f"[TextFileSelector] Error: File not found at path '{full_path}'.")
             self.file_content_cache[filename] = ""
             return ""
        except Exception as e:
            print(f"[TextFileSelector] Error reading file '{full_path}' with encoding '{encoding}': {e}")
            self.file_content_cache[filename] = ""
            return ""

    def select_and_read_file(self, folder_path, mode, seed, reset_state, cache_chunk_size, encoding, filename_filter):
        clean_folder_path = folder_path.strip()
        current_list_hash = self.last_list_hash
        needs_full_reset = False
        reset_reason = ""

        if clean_folder_path != self.last_folder_path:
            print(f"[TextFileSelector] Folder path changed to: '{clean_folder_path}'")
            self.last_folder_path = clean_folder_path
            if clean_folder_path and os.path.isdir(clean_folder_path):
                self.cached_file_list = self._scan_folder(clean_folder_path)
            else:
                self.cached_file_list = []
                if clean_folder_path:
                    print(f"[TextFileSelector] Warning: Invalid folder path provided: '{clean_folder_path}'")

            current_list_hash = hashlib.sha256(str(self.cached_file_list).encode()).hexdigest()
            if current_list_hash != self.last_list_hash:
                 needs_full_reset = True
                 reset_reason = "Folder path or file list content changed."

        if reset_state:
            needs_full_reset = True
            reset_reason = "Manual state reset requested."
        elif not self.cached_file_list and self.last_list_hash is not None:
            needs_full_reset = True
            reset_reason = "File list became empty."

        if needs_full_reset:
            print(f"[TextFileSelector] Full state reset triggered: {reset_reason}")
            self.round_robin_index = 0
            self.file_content_cache = {}
            self.cache_progress_index = 0
            self.last_list_hash = current_list_hash
            if not self.cached_file_list:
                 self.last_list_hash = None

        num_files = len(self.cached_file_list)
        processed_in_chunk = 0
        effective_cache_chunk_size = max(0, cache_chunk_size)
        if effective_cache_chunk_size > 0 and num_files > 0:
            for i in range(effective_cache_chunk_size):
                 if num_files == 0: break
                 idx_to_check = (self.cache_progress_index + i) % num_files
                 filename_to_check = self.cached_file_list[idx_to_check]
                 if filename_to_check not in self.file_content_cache:
                     self._read_and_cache_file(filename_to_check, self.last_folder_path, encoding)
                     processed_in_chunk += 1
            if num_files > 0:
                self.cache_progress_index = (self.cache_progress_index + effective_cache_chunk_size) % num_files

        chosen_filename = None
        clean_filename_filter = filename_filter.strip() 

        if not self.cached_file_list:
            print("[TextFileSelector] No text files found or folder path is invalid.")
            return ("", "None")
        else:
            effective_candidates = self.cached_file_list
            is_filtered = False
            if clean_filename_filter: 
                print(f"[TextFileSelector] Applying filename filter: '{clean_filename_filter}'")
                filtered_list_for_random = [f for f in self.cached_file_list if clean_filename_filter in f]
                if not filtered_list_for_random:
                    print(f"[TextFileSelector] Warning: No files match the filter '{clean_filename_filter}'.")
                else:
                     effective_candidates = filtered_list_for_random 
                     is_filtered = True 

            num_effective_candidates = len(effective_candidates)

            if num_effective_candidates == 0 and is_filtered:
                 print(f"[TextFileSelector] No files available after applying filter '{clean_filename_filter}'.")
                 return ("", "None")
            elif num_effective_candidates == 0 and not is_filtered:
                 print("[TextFileSelector] No files available.") 
                 return ("", "None")


            if mode == "random":
                random.seed(seed)
                chosen_filename = random.choice(effective_candidates)
                print(f"[TextFileSelector] Mode: random {'with filter' if is_filtered else ''} (Seed: {seed})")

            elif mode == "round-robin":
                num_total_files = len(self.cached_file_list) 
                found = False
                search_start_index = self.round_robin_index % num_total_files 

                for i in range(num_total_files): 
                    current_check_index = (search_start_index + i) % num_total_files
                    filename_to_check = self.cached_file_list[current_check_index]

                    filter_match = (not clean_filename_filter) or (clean_filename_filter in filename_to_check)

                    if filter_match: 
                        chosen_filename = filename_to_check
                        found_index = current_check_index
                        self.round_robin_index = found_index + 1
                        print(f"[TextFileSelector] Mode: round-robin {'with filter' if clean_filename_filter else ''} (Found at index {found_index}, Next RR Base: {self.round_robin_index})")
                        found = True
                        break

                if not found:
                    if clean_filename_filter:
                         print(f"[TextFileSelector] Mode: round-robin with filter '{clean_filename_filter}'. No matching file found.")
                    else:
                         print("[TextFileSelector] Mode: round-robin. No file found (List might be empty unexpectedly).")
                    return ("", "None") 

            else: 
                print(f"[TextFileSelector] Warning: Unknown mode '{mode}'. Falling back to random.")
                random.seed(seed)
                chosen_filename = random.choice(effective_candidates) 

            if chosen_filename:
                print(f"[TextFileSelector] Selected file: {chosen_filename}")
                file_content = self._read_and_cache_file(chosen_filename, self.last_folder_path, encoding)
                return (file_content, chosen_filename)
            else:
                print("[TextFileSelector] Internal error: Could not select a file.")
                return ("", "None")