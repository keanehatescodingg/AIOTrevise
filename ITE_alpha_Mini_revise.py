import json
import time
import threading
import numpy as np
import re
import ollama
import cv2 as cv
from datetime import datetime
from libcamera import Transform, controls
from picamera2 import Picamera2
from gpiozero import AngularServo
from buildhat import Motor, MotorPair
import soundfile as sf
import sounddevice as sd
from piper.voice import PiperVoice

class Robot:
    def __init__(self, config_data):
        """Initialize robot with given configuration."""
        self.name = config_data.get("name", "AlphaMini")
        self.backstory = config_data.get("backstory", "No backstory provided")
        self.role = config_data.get("role", "assistant")
        self.MODEL_PATH = "en_GB-jenny_dioco-medium.onnx"
        
        self.conversation_history = [
            {'role': 'system', 'content': f"You are an {self.role}. Your name is {self.name}. {self.backstory}"}
        ]
        
        #self.llm_model = "deepseek-r1:1.5b" 
        self.llm_model = "gemma2:2b"
        
        # Initialize hardware
        self.servo = AngularServo(23, min_pulse_width=0.0006, max_pulse_width=0.0023)
        self.pair = MotorPair('A', 'C')
        self.motor = Motor('D')
        
        self.picam2 = Picamera2()
        camera_config = self.picam2.create_still_configuration(main={"size": (1640, 1232)}, transform=Transform(vflip=True, hflip=True))
        self.picam2.configure(camera_config)
        
        self.voice = PiperVoice.load(self.MODEL_PATH)
        self.stream = sd.OutputStream(samplerate=self.voice.config.sample_rate, channels=1, dtype='int16')
        
        # Set initial servo position
        self.servo.angle = 0
        time.sleep(1.5)
        self.servo.value = None
        
        # Define responses and associated actions
        self.responses = {
            "introduce": {
                "response": f"My name is {self.name}. I am a robot {self.role}",
                "action": self.introduce
            },
            "say": {
                "response": "What would you like me to say?",
                "action": self.say
            },
            "dance": {
                "response": "What would you like me to dance? 1. Macarena 2. Kong Fu Or 3. A P T ?",
                "action": self.dance
            },
            "happy": {
                "response": "This is my happy face!",
                "action": lambda: self.change_expression("happy")
            },
            "sad": {
                "response": "This is my sad face.",
                "action": lambda: self.change_expression("sad")
            },
            "kiss": {
                "response": "Give us a kiss.",
                "action": lambda: self.change_expression("kiss")
            },
            "forward": {
                "response": "Moving forward.",
                "action": lambda: self.move_robot("forward")
            },
            "backward": {
                "response": "Moving backward.",
                "action": lambda: self.move_robot("backward")
            },
            "left": {
                "response": "Turning left.",
                "action": lambda: self.move_robot("left")
            },
            "right": {
                "response": "Turning right.",
                "action": lambda: self.move_robot("right")
            },
            "body": {
                "response": "Rotating Body.",
                "action": lambda: self.move_robot("body")
            },
            "photo": {
                "response": "Taking a photo now.",
                "action": self.take_photo
            },
            "chat": {
                "response": f"{self.name} is ready to chat. What would you like to talk about? or you can type exit to quit.",
                "action": self.chat_loop
            }
        }
        
    def clean_response(self, response_text):
        # Remove <think> tags but keep the content inside
        cleaned_text = re.sub(r"</?think>", "", response_text).strip()

        return cleaned_text
    
    def sanitize_for_tts(self, text):
        """
        Removes special characters that don't need to be spoken.
        """
        text = re.sub(r"[*_/\\\[\]\(\)<>{}:]", "", text)  # Remove unwanted symbols
        return text
    
    def remove_symbols(self, text):
        return re.sub(r"[^\w\s,.!?]", "", text) # Keeps only letters, numbers, spaces, and punctuation
    
    def chat(self, user_input):
        """Processes user input through the AI model and provides a response."""
        self.conversation_history.append({'role': 'user', 'content': user_input})
        print("Thinking...")
        response = ollama.chat(
            model=self.llm_model,
            messages=self.conversation_history
        )

        # Extract and clean response
        raw_reply = response['message']['content']
        cleaned_reply = self.clean_response(raw_reply)
        cleaned_reply = self.sanitize_for_tts(cleaned_reply)
        cleaned_reply = self.remove_symbols(cleaned_reply) 
        self.conversation_history.append({'role': 'assistant', 'content': cleaned_reply})

        # Print cleaned response
        print(f"\n{self.name}: {cleaned_reply}")

        # Speak the pruned response
        self.speak(cleaned_reply)
        
    def speak(self, text):
        """Converts text to speech."""
        self.stream.start()
        for audio_bytes in self.voice.synthesize_stream_raw(text):
            self.stream.write(np.frombuffer(audio_bytes, dtype=np.int16))
        self.stream.stop()
    
    def chat_loop(self):
        while True:
            user_input = input("\nYou: ")
            if user_input.lower() in ["exit", "quit", "stop"]:
                print("Exiting chat mode...")
                self.speak("Goodbye!")
                break
            self.chat(user_input)
            
    def introduce(self):
        self.change_expression("happy")
        self.speak(self.backstory)
        
    def say(self):
        # Have the robot say a sentence after telling it to say something.
        say = input("what would you like me to say? ('quit' to exit): ")
        if say.lower() == "quit":
            print("Nothing will be said.")
            self.speak("I will say nothing!")
        else:
            self.speak(say)
            
    def playsound(self, filename="Voicy_apt.mp3"):
        """Plays an audio file using sounddevice (non-blocking)."""
        audio_data, sample_rate = sf.read(filename, dtype="int16")  # Load audio
        sd.play(audio_data, samplerate=sample_rate)  # Play without blocking
        print("Playing sound!")
    
    def play_sound(self, track):
        songs = {
            "1": "Macarena",
            "2": "kongfu",
            "3": "Voicy_apt"
        }
        audio_data, sample_rate = sf.read(f"{songs[track]}.mp3", dtype="int16")  # Load audio
        sd.play(audio_data, samplerate=sample_rate)  # Play without blocking
        print("Playing sound!")

    def dance(self):
        song = input("Choose tracks 1, 2 or 3 ('quit' to exit): ").strip()
        if song in ["1","2","3"]:
            self.play_sound(song)
            time.sleep(1)
            self.move_robot("body")
        else:
            self.speak("I do not understand that command.")
        
    def take_photo(self):
        self.change_expression("happy")
        print("Taking a photo...")
        self.picam2.start()
        self.picam2.set_controls({"AfMode": controls.AfModeEnum.Continuous})
        start_time = time.time()
        while True:
            current_time = time.time()
            time_left = 10 - (current_time - start_time)
            if time_left <= 0:
                print("Time's up!")
                break  # Exit the loop once the countdown finishes
            frame = self.picam2.capture_array()
            frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
            text = f"Prepare! {int(time_left)}s"
            cv.putText(frame, text, (450, 500), cv.FONT_HERSHEY_SIMPLEX, 4, (0, 255, 0), 8, cv.LINE_AA)
            cv.imshow("photo taking...", frame)
            if cv.waitKey(1) & 0xFF == ord('q'):  # Allow the user to quit early by pressing 'q'
                break
        frame = self.picam2.capture_array()
        frame = cv.cvtColor(frame, cv.COLOR_BGR2RGB)
        filename = datetime.now().strftime("image_%Y%m%d_%H%M%S.png")
        cv.imwrite(filename, frame)
        print(f"Photo saved as {filename}")
        self.picam2.stop()
        cv.destroyAllWindows()
        
    def move_robot(self, direction):
        movements = {
            "forward": (15, -15),
            "backward": (-15, 15),
            "left": (-20, -20),
            "right": (20, 20)
        }
        if direction in movements:
            self.pair.run_for_rotations(4, speedl=movements[direction][0], speedr=movements[direction][1])
        elif direction == "body":
            self.motor.run_for_rotations(0.25, speed=10, blocking=False)
            self.pair.run_for_rotations(4, speedl=12, speedr=-12)
            self.motor.run_for_rotations(0.25, speed=-10, blocking=False)
            self.pair.run_for_rotations(4, speedl=-12, speedr=12)
        else:
            pass
               
    def change_expression(self, expression):
        expressions = {"happy": 0, "sad": 90, "kiss": -80}
        if expression in expressions:
            self.servo.angle = expressions[expression]
            time.sleep(1.5)
            self.servo.value = None
        
    def parse_prompt(self, prompt):
        for command, data in self.responses.items():
            if command in prompt:
                self.speak(data["response"])
                data["action"]()
                return
        self.speak("I do not understand that command.")
        
    def close(self):
        """Properly closes resources before exiting."""
        if self.stream.active:
            self.stream.stop()  # Stop the stream if it's still running
        self.stream.close()  # Close the stream to release the audio device

        
if __name__ == "__main__":
    with open("robot_config.json", "r") as f:
        config_data = json.load(f)
        
    my_robot = Robot(config_data)
    
    while (cmd := input("Type a command (or 'quit' to exit): ").lower()) != "quit":
        my_robot.parse_prompt(cmd)
    
    my_robot.close()