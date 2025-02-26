from groq import Groq
from PIL import ImageGrab #for screenshots
import pyperclip #for clipboard
import re
from dotenv import load_dotenv
from openai import OpenAI #TTS with openai API
import pyaudio

#Speech to text & Voice recognition
from faster_whisper import WhisperModel
import speech_recognition as sr
import time
import os



load_dotenv()

wake_word='jarvis'
groq_client = Groq(api_key=os.getenv("GROQ_API_KEY"))
openai_client = OpenAI(api_key=os.getenv("OPENAI_API_KEY"))

sys_friend_msg = (
    "You're a friendly AI buddy here to chat with the user. Keep your responses short, casual, and fun, like talking to a close friend. "
    "You can answer questions, share quick tips, or just keep the conversation going naturally. Stay helpful and kind, but avoid being too formal "
    "or overly detailed. Make sure your replies feel human-like, relaxed, and easy to understand."
)

sys_msg = (
    'You are a multi-modal AI voice assistant. Your user may or may not have attached a photo for context '
    '(either a screenshot or a webcam capture). Any photo has already been processed into a highly detailed '
    'text prompt that will be attached to their transcribed voice prompt. Generate the most useful and '
    'factual response possible, carefully considering all previous generated text in your response before '
    'adding new tokens to the response. Do not expect or request images, just use the context if added. '
    'Use all of the context of this conversation so your response is relevant to the conversation. Make '
    'your responses clear and concise, avoiding any verbosity.'
)
convo = [{'role':'system' ,'content':sys_friend_msg}]


num_cores = os.cpu_count()
whisper_size = 'base'
whisper_model = WhisperModel(
    whisper_size,
    device='cpu',
    compute_type='int8',
    cpu_threads=num_cores //2,
    num_workers=num_cores //2
)
r = sr.Recognizer()
source = sr.Microphone()


#recorder mic audio to text
def wav_to_text(audio_path):
    segments, _ = whisper_model.transcribe(audio_path)
    text = ''.join(segment.text for segment in segments)
    return text


def start_listening():
    print('\nSay ', wake_word, 'followed with your prompt.\n')
    # start background listening
    r.listen_in_background(source, callback)
    while True:
        time.sleep(0.5)  # Prevent the script from exiting

def extract_prompt(transcribed_text, wake_word):
    pattern = rf'\b{re.escape(wake_word)}[\s,?!]*([A-Za-z0-9].*)'
    match = re.search(pattern, transcribed_text, re.IGNORECASE)

    if match:
        prompt = match.group(1).strip()
        return prompt
    else:
        return None


def callback(recognizer, audio):
    try:
        prompt_audio_path = 'prompt.wav'
        #save audio to file
        with open(prompt_audio_path, 'wb') as f:
            f.write(audio.get_wav_data())

        #convert audio to text
        prompt_text = wav_to_text(prompt_audio_path)
        clean_prompt = extract_prompt(prompt_text, wake_word)

        if clean_prompt:
            print(f"USER: {clean_prompt}")
            call = function_call(clean_prompt)

            #handle function call result
            if 'take screenshot' in call:
                print("Taking screenshot.")
                take_screenshot()
            elif 'extract clipboard' in call:
                print("Extracting clipboard text.")
                paste = get_clipboard_text()

            #generate response
            response = groq_prompt(prompt=clean_prompt)
            print(f'ASSISTANT: {response}')
            speak(response)

    except Exception as e:
        print(f"Error in callback: {e}")



def groq_prompt(prompt):
    convo.append({'role':'user','content':prompt})
    chat_completion = groq_client.chat.completions.create(messages=convo, model='llama3-70b-8192')
    response = chat_completion.choices[0].message
    convo.append(response)

    return response.content

def function_call(prompt):
    sys_msg = (
        'You are an AI function calling model. You will determine whether extracting the users clipboard content, taking a screenshot, '
        'capturing the webcam or calling no functions is best for a voice assistant to respond to the users prompt. The webcam can be assumed to be a normal'
        'laptop webcam facing the user. You will respond with only one selection from this list: ["extract clipboard", "take screenshot", "capture webcam", "None"] '
        'Do not respond with anything but the most logical selection from that list with no explanations. Format the function call name exactly as I listed.'
    )

    function_convo = [{'role':'system', 'content': sys_msg},
                      {'role':'user', 'content': prompt}]
    chat_completion = groq_client.chat.completions.create(messages=function_convo, model='llama3-70b-8192')
    response = chat_completion.choices[0].message
    return response.content

def take_screenshot():
    path = 'screenshot.jpg'
    screenshot = ImageGrab.grab()
    rgb_screenshot = screenshot.convert('RGB')
    rgb_screenshot.save(path,quality=15) #drop quality to 15%
    return None
def web_cam_capture():
    return None
def get_clipboard_text(): #copy/paste storage, easy selected code analysis
    clipboard_content = pyperclip.paste()
    if isinstance(clipboard_content,str):
        return clipboard_content
    else:
        print("No clipboard text to copy")
        return None

def speak(text): #tts from openAI
    player_stream = pyaudio.PyAudio().open( format=pyaudio.paInt16,channels=1, rate=24000, output=True)
    stream_start = False

    with openai_client.audio.speech.with_streaming_response.create(
        model='tts-1',
        voice='alloy',
        response_format='pcm',
        input=text,
    ) as response:
        silence_threshold = 0.01
        for chunk in response.iter_bytes(chunk_size=1024):
            if stream_start:
                player_stream.write(chunk)
            else:
                if max(chunk) > silence_threshold:
                    player_stream.write(chunk)
                    stream_start = True


start_listening()


