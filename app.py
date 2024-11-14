import customtkinter as ctk
from tkinter import filedialog, messagebox
import pandas as pd
import os
import threading
from moviepy.editor import VideoFileClip
import speech_recognition as sr
from pydub import AudioSegment
from pydub.silence import split_on_silence
import tempfile
from langchain_google_genai import GoogleGenerativeAI
from langchain.prompts import PromptTemplate
from langchain.chains import LLMChain
import re

class VideoAnalysisApp:
    def __init__(self):
        self.setup_window()
        self.setup_variables()
        self.create_widgets()
        self.srt_path = None
        
    def setup_window(self):
        self.root = ctk.CTk()
        self.root.title("Video Content Analyzer")
        self.root.geometry("1200x800")
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
    def setup_variables(self):
        self.video_path = ctk.StringVar()
        self.status_var = ctk.StringVar(value="Ready to start...")
        self.api_key = 'YOUR_GEMINI_API_KEY'
        
    def create_widgets(self):
        # Main container
        self.main_frame = ctk.CTkFrame(self.root)
        self.main_frame.pack(fill="both", expand=True, padx=20, pady=20)
        
        # File Selection Area
        self.create_file_selection_area()
        
        # Status Area
        self.create_status_area()
        
        # Process Buttons
        self.create_process_buttons()
        
        # Results Viewer
        self.create_results_viewer()
        
    def create_file_selection_area(self):
        file_frame = ctk.CTkFrame(self.main_frame)
        file_frame.pack(fill="x", padx=10, pady=10)
        
        ctk.CTkLabel(file_frame, text="Video File:").pack(side="left", padx=5)
        
        self.file_entry = ctk.CTkEntry(file_frame, textvariable=self.video_path, width=400)
        self.file_entry.pack(side="left", padx=5, fill="x", expand=True)
        
        browse_btn = ctk.CTkButton(file_frame, text="Browse", command=self.browse_video)
        browse_btn.pack(side="right", padx=5)
        
    def create_status_area(self):
        status_frame = ctk.CTkFrame(self.main_frame)
        status_frame.pack(fill="x", padx=10, pady=10)
        
        self.progress_bar = ctk.CTkProgressBar(status_frame)
        self.progress_bar.pack(fill="x", padx=10, pady=5)
        self.progress_bar.set(0)
        
        status_label = ctk.CTkLabel(status_frame, textvariable=self.status_var)
        status_label.pack(pady=5)
        
    def create_process_buttons(self):
        btn_frame = ctk.CTkFrame(self.main_frame)
        btn_frame.pack(fill="x", padx=10, pady=10)
        
        generate_btn = ctk.CTkButton(btn_frame, text="Generate Subtitles", 
                                   command=lambda: self.run_in_thread(self.generate_subtitles))
        generate_btn.pack(side="left", padx=5, expand=True)
        
        analyze_btn = ctk.CTkButton(btn_frame, text="Analyze Content", 
                                  command=lambda: self.run_in_thread(self.analyze_content))
        analyze_btn.pack(side="left", padx=5, expand=True)
        
        save_btn = ctk.CTkButton(btn_frame, text="Save Analysis", 
                                command=self.save_analysis)
        save_btn.pack(side="left", padx=5, expand=True)
        
    def create_results_viewer(self):
        self.results_frame = ctk.CTkFrame(self.main_frame)
        self.results_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create a tabview for different sections of the analysis
        self.tabview = ctk.CTkTabview(self.results_frame)
        self.tabview.pack(fill="both", expand=True, padx=5, pady=5)
        
        # Add tabs
        self.tabs = {
            "Overview": self.tabview.add("Overview"),
            "Key Points": self.tabview.add("Key Points"),
            "Quotes": self.tabview.add("Quotes"),
            "Technical": self.tabview.add("Technical")
        }
        
        # Add text widgets for each tab
        self.tab_textboxes = {}
        for tab_name, tab in self.tabs.items():
            textbox = ctk.CTkTextbox(tab, wrap="word")
            textbox.pack(fill="both", expand=True, padx=5, pady=5)
            self.tab_textboxes[tab_name] = textbox
            
    def browse_video(self):
        filename = filedialog.askopenfilename(
            filetypes=[("Video files", "*.mp4 *.avi *.mkv *.mov")])
        if filename:
            self.video_path.set(filename)
            
    def run_in_thread(self, func):
        thread = threading.Thread(target=func)
        thread.daemon = True
        thread.start()
        
    def update_progress(self, value):
        self.progress_bar.set(value)
        self.root.update_idletasks()
        
    def update_status(self, message):
        self.status_var.set(message)
        self.root.update_idletasks()
        
    def generate_subtitles(self):
        if not self.video_path.get():
            messagebox.showerror("Error", "Please select a video file first.")
            return
            
        self.update_status("Generating subtitles...")
        self.update_progress(0.2)
        
        try:
            video_path = self.video_path.get()
            self.srt_path = os.path.splitext(video_path)[0] + '.srt'
            
            # Extract audio
            temp_audio_path = 'temp_audio.wav'
            video = VideoFileClip(video_path)
            video.audio.write_audiofile(temp_audio_path)
            video.close()
            
            self.update_progress(0.4)
            
            # Process audio
            audio = AudioSegment.from_wav(temp_audio_path)
            chunks = split_on_silence(audio, min_silence_len=500, silence_thresh=-40)
            
            self.update_progress(0.6)
            
            # Generate subtitles
            recognizer = sr.Recognizer()
            with open(self.srt_path, 'w', encoding='utf-8') as srt_file:
                for i, chunk in enumerate(chunks, 1):
                    with tempfile.NamedTemporaryFile(suffix='.wav') as temp_chunk:
                        chunk.export(temp_chunk.name, format='wav')
                        with sr.AudioFile(temp_chunk.name) as source:
                            audio_data = recognizer.record(source)
                            try:
                                text = recognizer.recognize_google(audio_data)
                                start_time = sum(len(c) for c in chunks[:i-1])
                                end_time = start_time + len(chunk)
                                
                                srt_file.write(f"{i}\n")
                                srt_file.write(f"{self.format_timestamp(start_time)} --> {self.format_timestamp(end_time)}\n")
                                srt_file.write(f"{text}\n\n")
                            except:
                                continue
            
            self.update_progress(1.0)
            self.update_status("Subtitles generated successfully!")
            
            # Clean up
            os.remove(temp_audio_path)
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to generate subtitles: {str(e)}")
            self.update_status("Error generating subtitles")
            
    def format_timestamp(self, milliseconds):
        seconds = milliseconds // 1000
        remaining_ms = milliseconds % 1000
        hours = seconds // 3600
        minutes = (seconds % 3600) // 60
        seconds = seconds % 60
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{remaining_ms:03d}"
        
    def analyze_content(self):
        if not self.srt_path or not os.path.exists(self.srt_path):
            messagebox.showerror("Error", "Please generate subtitles first.")
            return
            
        if not self.api_key:
            messagebox.showerror("Error", "Please set GOOGLE_API_KEY environment variable.")
            return
            
        self.update_status("Analyzing content...")
        self.update_progress(0.3)
        
        try:
            # Clean SRT content
            with open(self.srt_path, 'r', encoding='utf-8') as file:
                content = file.read()
            """
            Clean and extract text content from SRT file
            """
            
            # Remove SRT indices and timestamps
            pattern = r'\d+\n\d{2}:\d{2}:\d{2},\d{3} --> \d{2}:\d{2}:\d{2},\d{3}\n'
            cleaned_content = re.sub(pattern, '', content)
            
            # Remove empty lines and clean up spacing
            cleaned_content = '\n'.join(line.strip() for line in cleaned_content.split('\n') if line.strip())
            
            self.update_progress(0.5)
            
            # Initialize Gemini and run analysis
            llm = GoogleGenerativeAI(model="gemini-1.5-flash", google_api_key=self.api_key, temperature=0.1)
            
            chain = LLMChain(llm=llm, prompt=self.create_analysis_prompt())
            response = chain.run(text=cleaned_content)
            try:
                # Clean up the response string to ensure it's valid Python syntax
                response = response.replace("```python", "").replace("```", "").strip()
                result_dict = eval(response)
                
                # Ensure all expected keys are present
                expected_keys = [
                    'main_topic', 'speakers', 'key_points', 'statistics',
                    'recommendations', 'conclusions', 'notable_quotes',
                    'technical_terms', 'challenges', 'solutions'
                ]
                
                for key in expected_keys:
                    if key not in result_dict:
                        result_dict[key] = [] if key in ['key_points', 'statistics', 'recommendations', 
                                                    'notable_quotes', 'technical_terms', 'challenges', 
                                                    'solutions'] else ""
                
            except Exception as e:
                print(f"Error parsing LLM response: {e}")
                return {key: [] if key in ['key_points', 'statistics', 'recommendations', 
                                        'notable_quotes', 'technical_terms', 'challenges', 
                                        'solutions'] else "" for key in expected_keys}
            
            
            
            
            self.update_progress(0.8)
            
            # Parse and display results
            self.analysis_results = result_dict
            self.display_results()
            
            self.update_progress(1.0)
            self.update_status("Analysis complete!")
            
        except Exception as e:
            messagebox.showerror("Error", f"Failed to analyze content: {str(e)}")
            self.update_status("Error analyzing content")
            
    def create_analysis_prompt(self):
        template = """
        Analyze the following video transcript and answer these specific questions.
        Please provide detailed, factual responses based solely on the content provided.
        
        TRANSCRIPT:
        {text}
        
        Please analyze the content and answer the following questions:
        1. What is the main topic or theme of this video?
        2. Who are the key speakers or participants mentioned (if any)?
        3. What are the 3-5 most important points discussed?
        4. Are there any significant statistics or numerical data mentioned?
        5. What practical advice or recommendations are given (if any)?
        6. What are the key conclusions or takeaways?
        7. Are there any notable quotes or memorable statements?
        8. What technical terms or specialized vocabulary are used?
        9. What problems or challenges are discussed?
        10. What solutions or resolutions are proposed?

        Format your response as a Python dictionary with these exact keys:
        - main_topic
        - speakers
        - key_points
        - statistics
        - recommendations
        - conclusions
        - notable_quotes
        - technical_terms
        - challenges
        - solutions
        """
        return PromptTemplate(input_variables=["text"], template=template)
        
    def display_results(self):
        if not hasattr(self, 'analysis_results'):
            return
            
        # Clear previous results
        for textbox in self.tab_textboxes.values():
            textbox.delete("1.0", "end")
            
        # Overview Tab
        overview_text = f"""
        Main Topic: {self.analysis_results['main_topic']}
        
        Speakers: {', '.join(self.analysis_results['speakers']) if isinstance(self.analysis_results['speakers'], list) else self.analysis_results['speakers']}
        
        Conclusions:
        {self.analysis_results['conclusions']}
        """
        self.tab_textboxes["Overview"].insert("1.0", overview_text)
        
        # Key Points Tab
        key_points_text = "Key Points:\n\n"
        for i, point in enumerate(self.analysis_results['key_points'], 1):
            key_points_text += f"{i}. {point}\n\n"
        key_points_text += "\nRecommendations:\n\n"
        for i, rec in enumerate(self.analysis_results['recommendations'], 1):
            key_points_text += f"{i}. {rec}\n\n"
        self.tab_textboxes["Key Points"].insert("1.0", key_points_text)
        
        # Quotes Tab
        quotes_text = "Notable Quotes:\n\n"
        for i, quote in enumerate(self.analysis_results['notable_quotes'], 1):
            quotes_text += f"{i}. {quote}\n\n"
        self.tab_textboxes["Quotes"].insert("1.0", quotes_text)
        
        # Technical Tab
        technical_text = "Technical Terms:\n\n"
        for i, term in enumerate(self.analysis_results['technical_terms'], 1):
            technical_text += f"{i}. {term}\n\n"
        technical_text += "\nChallenges & Solutions:\n\n"
        for i, (challenge, solution) in enumerate(zip(self.analysis_results['challenges'], 
                                                    self.analysis_results['solutions']), 1):
            technical_text += f"Challenge {i}: {challenge}\n"
            technical_text += f"Solution {i}: {solution}\n\n"
        self.tab_textboxes["Technical"].insert("1.0", technical_text)
        
    def save_analysis(self):
        if not hasattr(self, 'analysis_results'):
            messagebox.showerror("Error", "Please analyze content first.")
            return
            
        file_path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv")]
        )
        
        if file_path:
            try:                
                # Save to CSV
                df = pd.DataFrame([self.analysis_results])
                df.to_csv(file_path, index=False)
                messagebox.showinfo("Success", f"Analysis saved to {file_path}")
                
            except Exception as e:
                messagebox.showerror("Error", f"Failed to save analysis: {str(e)}")
                
    def run(self):
        self.root.mainloop()

if __name__ == "__main__":
    app = VideoAnalysisApp()
    app.run()