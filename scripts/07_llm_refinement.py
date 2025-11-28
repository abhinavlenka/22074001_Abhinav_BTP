#!/usr/bin/env python3
"""
LLM-based Few-Shot Refinement
Final post-processing using Google Gemini or other LLM APIs
"""

import os
import time
from pathlib import Path
from tqdm import tqdm

# Try multiple LLM options
try:
    import google.generativeai as genai
    GEMINI_AVAILABLE = True
except ImportError:
    GEMINI_AVAILABLE = False

try:
    from openai import OpenAI
    OPENAI_AVAILABLE = True
except ImportError:
    OPENAI_AVAILABLE = False

class LLMRefiner:
    """Refine MT output using LLM few-shot learning"""

    def __init__(self, api_provider="gemini", api_key="AIzaSyBkf8ZMaxLNIqcY0IGK_jpoNbheYjZ0Xsk", verbose=True):
        self.api_provider = api_provider
        self.api_key = api_key or os.getenv("GEMINI_API_KEY") or os.getenv("OPENAI_API_KEY")
        self.verbose = verbose
        
        self.few_shot_examples = [
            {
                "english": "Hello, how are you?",
                "initial_translation": "नमस्कार, तुम कैसे हो?",
                "refined_translation": "नमस्ते, आप कैसे हैं?"
            },
            {
                "english": "I am going to the market.",
                "initial_translation": "मैं बाज़ार को जा रहा हूँ।",
                "refined_translation": "मैं बाज़ार जा रहा हूँ।"
            },
            {
                "english": "This is very important.",
                "initial_translation": "यह बहुत जरूरी है।",
                "refined_translation": "यह बहुत महत्वपूर्ण है।"
            },
            {
                "english": "Thank you for your help.",
                "initial_translation": "तुम्हारी मदद के लिए धन्यवाद।",
                "refined_translation": "आपकी सहायता के लिए धन्यवाद।"
            }
        ]
        
        if api_provider == "gemini" and GEMINI_AVAILABLE and self.api_key:
            genai.configure(api_key=self.api_key)
            self.model = genai.GenerativeModel('gemini-2.5-flash')
            if self.verbose:
                print("Using Google Gemini API")
        elif api_provider == "openai" and OPENAI_AVAILABLE and self.api_key:
            self.client = OpenAI(api_key=self.api_key)
            if self.verbose:
                print("Using OpenAI API")
        else:
            if self.verbose:
                print("Warning: No LLM API configured. Using rule-based refinement.")
            self.model = None
    
    def create_prompt(self, english_text, mt_output):
        """Create few-shot prompt for refinement"""
        prompt = """You are an expert English-Hindi translator. Your task is to refine machine translation outputs to make them more natural, grammatically correct, and culturally appropriate.

Here are some examples of refinements:

"""
        for ex in self.few_shot_examples:
            prompt += f"English: {ex['english']}\n"
            prompt += f"Initial MT: {ex['initial_translation']}\n"
            prompt += f"Refined: {ex['refined_translation']}\n\n"
        
        prompt += f"""Now refine the following translation:

English: {english_text}
Initial MT: {mt_output}
Refined:"""
        
        return prompt
    
    def refine_with_gemini(self, english_text, mt_output):
        """Refine using Gemini API"""
        try:
            prompt = self.create_prompt(english_text, mt_output)
            response = self.model.generate_content(prompt)
            return response.text.strip()
        except Exception as e:
            print(f"Gemini API error: {e}")
            return mt_output
    
    def refine_with_openai(self, english_text, mt_output):
        """Refine using OpenAI API"""
        try:
            prompt = self.create_prompt(english_text, mt_output)
            
            response = self.client.chat.completions.create(
                model="gpt-3.5-turbo",
                messages=[
                    {"role": "system", "content": "You are an expert English-Hindi translator."},
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=200
            )
            
            return response.choices[0].message.content.strip()
        except Exception as e:
            print(f"OpenAI API error: {e}")
            return mt_output
    
    def rule_based_refinement(self, mt_output):
        """Simple rule-based refinement as fallback"""
        # Basic post-processing rules
        refined = mt_output
        
        # Common fixes
        replacements = {
            "तुम": "आप",  # Formal you
            "तेरा": "आपका",  # Formal your
            "कर रहा है": "कर रहे हैं",  # Formal present continuous
        }
        
        for old, new in replacements.items():
            refined = refined.replace(old, new)
        
        return refined
    
    def refine(self, english_text, mt_output):
        """Refine a single translation"""
        if self.model is None:
            return self.rule_based_refinement(mt_output)
        
        if self.api_provider == "gemini":
            return self.refine_with_gemini(english_text, mt_output)
        elif self.api_provider == "openai":
            return self.refine_with_openai(english_text, mt_output)
        else:
            return self.rule_based_refinement(mt_output)
    
    def refine_file(self, english_file, mt_file, output_file, delay=1.0):
        """Refine translations from files"""
        print(f"\nRefining translations...")
        print(f"Input: {mt_file}")
        print(f"Output: {output_file}")
        
        with open(english_file, 'r', encoding='utf-8') as f:
            english_lines = [line.strip() for line in f if line.strip()]
        
        with open(mt_file, 'r', encoding='utf-8') as f:
            mt_lines = [line.strip() for line in f if line.strip()]
        
        refined_lines = []
        
        for eng, mt in tqdm(zip(english_lines, mt_lines), total=len(english_lines)):
            refined = self.refine(eng, mt)
            refined_lines.append(refined)
            
            # Rate limiting for API calls
            if self.model is not None:
                time.sleep(delay)
        
        with open(output_file, 'w', encoding='utf-8') as f:
            f.write('\n'.join(refined_lines))
        
        print(f"Refined translations saved to: {output_file}")

def main():
    import argparse
    
    parser = argparse.ArgumentParser(description="Refine MT output with LLM")
    parser.add_argument("--english", required=True, help="English source file")
    parser.add_argument("--mt_output", required=True, help="MT output file to refine")
    parser.add_argument("--output", required=True, help="Output file for refined translations")
    parser.add_argument("--api", choices=["gemini", "openai"], default="gemini",
                       help="LLM API to use")
    parser.add_argument("--api_key", help="API key (or set GEMINI_API_KEY/OPENAI_API_KEY env var)")
    parser.add_argument("--delay", type=float, default=1.0, 
                       help="Delay between API calls (seconds)")
    
    args = parser.parse_args()
    
    refiner = LLMRefiner(api_provider=args.api, api_key=args.api_key)
    refiner.refine_file(args.english, args.mt_output, args.output, delay=args.delay)

if __name__ == "__main__":
    main()
