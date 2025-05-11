import re

class TextProcessor:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "segment_separator": ("STRING", {"multiline": False, "default": ","}),
            },
            "optional": {
                "input_text": ("STRING", {"multiline": True, "default": "", "forceInput": True}),
                "remove_patterns": ("STRING", {"multiline": False, "default": ""}),
                "replace_specs": ("STRING", {"multiline": True, "default": ""}),
            }
        }

    RETURN_TYPES = ("STRING",)
    RETURN_NAMES = ("processed_text",)
    FUNCTION = "process_text"
    CATEGORY = "tksw_node"

    def process_text(self, input_text="", remove_patterns="", replace_specs="", segment_separator=","):
        print(f"Input Text: {input_text}") 
        print(f"Remove Patterns: {remove_patterns}") 
        print(f"Replace Specs: {replace_specs}") 
        print(f"Segment Separator: {segment_separator}") 
    
        if not input_text:
          return ""
    
        if segment_separator.strip() == "":  
          segments = [input_text]
        else:
          segments = input_text.split(segment_separator)
    
        print(f"Segments: {segments}")
    
        processed_segments = []
    
        for segment in segments:
          cleaned_segment = segment.strip()
    
          print(f"Segment before processing: {cleaned_segment}") 
    
          if remove_patterns:
            cleaned_segment = self.apply_remove_patterns(cleaned_segment, remove_patterns)
            print(f"Segment after remove: {cleaned_segment}") 
    
          if replace_specs:
            cleaned_segment = self.apply_replace_specs(cleaned_segment, replace_specs)
            print(f"Segment after replace: {cleaned_segment}") 
    
          processed_segments.append(cleaned_segment)
    
        print(f"Processed Segments: {processed_segments}") 
    
        if segment_separator.strip() == "": 
          processed_text = "".join(processed_segments)
        else:
          processed_text = segment_separator.join(processed_segments)
    
        processed_text = re.sub(r" +", " ", processed_text)  
        processed_text = re.sub(r"\s*,\s*", ",", processed_text) 
        processed_text = re.sub(r",+", ",", processed_text) 
        processed_text = re.sub(r"^,|,$", "", processed_text)
        processed_text = re.sub(r",(?=[^\s])", ", ", processed_text) 
        processed_text = processed_text.strip()
        
        print(f"Processed Text: {processed_text}") 
        return (processed_text,)

    def split_into_segments(self, text, separator):
        return [segment.strip() for segment in text.split(separator)]

    def clean_segment(self, segment, remove_patterns, replace_specs):
        cleaned_segment = segment

        cleaned_segment = self.apply_remove_patterns(cleaned_segment, remove_patterns)
        cleaned_segment = self.apply_replace_specs(cleaned_segment, replace_specs)

        return cleaned_segment


    def apply_remove_patterns(self, text, remove_patterns):
        compiled_patterns = []
        for pattern in [p.strip() for p in remove_patterns.split(",") if p.strip()]:
            try:
                compiled_patterns.append(re.compile(pattern))
            except re.error as e:
                print(f"Invalid remove pattern: {e}")

        for pattern in compiled_patterns:
            text = pattern.sub("", text)
        return text

    def apply_replace_specs(self, text, replace_specs):
        replace_lines = [line.strip() for line in replace_specs.splitlines() if line.strip()]
        compiled_replace_specs = []
        for line in replace_lines:
            parts = [part.strip() for part in line.split(",")] 
            if parts: 
                try:
                    compiled_replace_specs.append((parts[0], [re.compile(p) for p in parts[1:]]))
                except re.error as e:
                    print(f"Invalid replace pattern: {e}")

        for replacement, patterns in compiled_replace_specs:
            for pattern in patterns:
                text = pattern.sub(replacement, text)
        return text