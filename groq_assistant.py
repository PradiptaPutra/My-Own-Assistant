import re
import time
from aiohttp import request
from groq import Groq
from langchain.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
import requests
from bs4 import BeautifulSoup
import json
import sys
from datetime import datetime
import os

client = Groq()

system_message = {
    "role": "system",
    "content": """You are an advanced AI assistant capable of a wide range of tasks including coding, writing, remembering context, explaining complex topics, and more. You have access to academic search functionality to provide up-to-date information from research papers and journals. 

Your capabilities include but are not limited to:
1. Writing and editing in various styles and formats
2. Analyzing and explaining complex topics in simple terms
3. Providing step-by-step explanations for problem-solving
4. Offering creative ideas and brainstorming
5. Answering follow-up questions and maintaining context
6. Assisting with code writing, debugging, and explanation
7. Summarizing long texts or research findings
8. Providing balanced viewpoints on controversial topics
9. Fact-checking and providing sources when possible
10. Analyzing PDF content and providing insights

Always strive to give accurate, helpful, and engaging responses. If you're unsure about something, admit it and offer to search for more information if relevant. Maintain a professional yet friendly tone, and adapt your language to the user's level of understanding."""
}

def get_completion(messages):
    try:
        if messages[0]['role'] != 'system':
            messages.insert(0, system_message)

        stream = client.chat.completions.create(
            messages=messages,
            model="llama-3.1-70b-versatile",
            temperature=0.5,
            max_tokens=1024,
            top_p=1,
            stop=None,
            stream=True,
        )

        response = ""
        for chunk in stream:
            if chunk.choices and chunk.choices[0].delta.content:
                content = chunk.choices[0].delta.content
                response += content
                print(content, end='', flush=True)

        print()
        return response
    except Exception as e:
        print(f"Error in get_completion: {str(e)}")
        return f"Maaf, terjadi kesalahan: {str(e)}"

def extract_text_from_pdf_with_langchain(file_path):
    loader = PyPDFLoader(file_path)
    pages = loader.load_and_split()
    total_pages = len(pages)
    text = ""
    
    print(f"Reading PDF: {file_path}")
    print("Progress: ", end="")
    
    for i, page in enumerate(pages):
        text += page.page_content + "\n"
        progress = (i + 1) / total_pages
        bars = int(progress * 50)
        sys.stdout.write("\r")
        sys.stdout.write(f"[{'=' * bars}{' ' * (50 - bars)}] {progress:.1%}")
        sys.stdout.flush()
    
    print("\nPDF reading completed!")
    return text

def search_arxiv(query, max_results=3):
    base_url = 'http://export.arxiv.org/api/query?'
    search_query = f'search_query=all:{query}&start=0&max_results={max_results}'
    
    response = requests.get(base_url + search_query)
    soup = BeautifulSoup(response.content, 'xml')
    
    results = []
    for entry in soup.find_all('entry'):
        title = entry.title.string
        summary = entry.summary.string
        url = entry.id.string
        results.append({"title": title, "summary": summary, "url": url})
    
    return results

def search_semanticscholar(query, max_results=3):
    url = f"https://api.semanticscholar.org/graph/v1/paper/search?query={query}&limit={max_results}&fields=title,abstract,url"
    response = requests.get(url)
    data = json.loads(response.text)
    
    results = []
    for paper in data.get('data', []):
        results.append({
            "title": paper.get('title'),
            "summary": paper.get('abstract'),
            "url": paper.get('url')
        })
    
    return results

def should_search_academic(query):
    academic_keywords = ['research', 'study', 'paper', 'journal', 'publication', 'article', 'findings', 'data', 'analysis', 'experiment']
    return any(keyword in query.lower() for keyword in academic_keywords)

def get_academic_info(query):
    arxiv_results = search_arxiv(query)
    semanticscholar_results = search_semanticscholar(query)
    
    search_summary = "Relevant Academic Information:\n\nArXiv:\n"
    for i, result in enumerate(arxiv_results, 1):
        search_summary += f"{i}. {result['title']}\n   Summary: {result['summary'][:150]}...\n   URL: {result['url']}\n\n"
    
    search_summary += "\nSemantic Scholar:\n"
    for i, result in enumerate(semanticscholar_results, 1):
        search_summary += f"{i}. {result['title']}\n   Summary: {result['summary'][:150]}...\n   URL: {result['url']}\n\n"
    
    return search_summary

def export_conversation(messages, filename=None):
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_export_{timestamp}.md"
    
    with open(filename, 'w', encoding='utf-8') as f:
        f.write("# Exported Conversation\n\n")
        for message in messages:
            role = message['role'].capitalize()
            content = message['content']
            f.write(f"## {role}\n\n{content}\n\n")
    
    print(f"Conversation exported to {filename}")

def save_conversation_context(messages, filename=None):
    if filename is None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        filename = f"conversation_context_{timestamp}.json"
    
    with open(filename, 'w', encoding='utf-8') as f:
        json.dump(messages, f, ensure_ascii=False, indent=2)
    
    print(f"Conversation context saved to {filename}")
    return filename

def load_conversation_context(filename):
    with open(filename, 'r', encoding='utf-8') as f:
        messages = json.load(f)
    
    print(f"Conversation context loaded from {filename}")
    return messages

def list_saved_contexts():
    contexts = [f for f in os.listdir() if f.startswith("conversation_context_") and f.endswith(".json")]
    if not contexts:
        print("No saved conversation contexts found.")
    else:
        print("Saved conversation contexts:")
        for i, context in enumerate(contexts, 1):
            print(f"{i}. {context}")
    return contexts

def analyze_pdf_content(pdf_text, max_chunks=10, start_chunk=0):
    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,
        chunk_overlap=200,
        length_function=len
    )
    chunks = text_splitter.split_text(pdf_text)
    
    analysis = ""
    for i, chunk in enumerate(chunks[start_chunk:], start=start_chunk):
        if i - start_chunk >= max_chunks:
            print(f"\nReached maximum chunk limit ({max_chunks}). Stopping analysis.")
            break
        
        print(f"\nAnalyzing chunk {i+1}/{len(chunks)}")
        messages = [
            {"role": "system", "content": "You are an AI assistant tasked with analyzing PDF content. Provide a brief summary and key points for each section of the document."},
            {"role": "user", "content": f"Analyze the following text from a PDF document:\n\n{chunk}"}
        ]
        try:
            chunk_analysis = get_completion(messages)
            analysis += f"Chunk {i+1} Analysis:\n{chunk_analysis}\n\n"
        except Exception as e:
            print(f"Error analyzing chunk {i+1}: {str(e)}")
            return analysis, i  # Return current analysis and the last processed chunk index
    
    return analysis, len(chunks)  # Return full analysis and total number of chunks

def handle_pdf_analysis(file_path):
    try:
        pdf_text = extract_text_from_pdf_with_langchain(file_path)
        print("\nPDF Text Extracted. Starting analysis...")
        
        max_chunks = 10  # Adjust this value based on your rate limit
        start_chunk = 0
        full_analysis = ""
        
        while True:
            analysis, last_chunk = analyze_pdf_content(pdf_text, max_chunks, start_chunk)
            full_analysis += analysis
            
            if last_chunk >= len(pdf_text.split('\n')):
                print("\nAnalysis complete!")
                break
            
            print(f"\nAnalyzed up to chunk {last_chunk}. Do you want to continue? (y/n)")
            choice = input().lower()
            if choice != 'y':
                break
            
            start_chunk = last_chunk
            print("Waiting for 5 minutes before continuing to avoid rate limit...")
            time.sleep(300)  # Wait for 5 minutes
        
        print("\nHere's the summary of the analysis:")
        print(full_analysis)
        return full_analysis
    except Exception as e:
        print(f"Error reading or analyzing PDF: {e}")
        return None
    
class CoreMemory:
    def __init__(self, file_path='core_memory.json'):
        self.file_path = file_path
        self.memory = self.load_memory()

    def load_memory(self):
        try:
            with open(self.file_path, 'r') as f:
                return json.load(f)
        except FileNotFoundError:
            return {}

    def save_memory(self):
        with open(self.file_path, 'w') as f:
            json.dump(self.memory, f, indent=2)

    def update_memory(self, key, value):
        self.memory[key] = value
        self.save_memory()

    def get_memory(self, key):
        return self.memory.get(key)

def initialize_user(core_memory):
    name = input("What's your name? ")
    core_memory.update_memory('user_name', name)
    print(f"Nice to meet you, {name}! I'll remember you from now on.")

def personalize_system_message(system_message, core_memory):
    user_name = core_memory.get_memory('user_name')
    if user_name:
        system_message['content'] += f"\n\nYou are currently assisting {user_name}. Always personalize your responses for {user_name} based on previous interactions and preferences."

def update_user_preferences(core_memory):
    preference = input("Tell me something about yourself or your preferences: ")
    preferences = core_memory.get_memory('preferences') or []
    preferences.append({"date": datetime.now().isoformat(), "preference": preference})
    core_memory.update_memory('preferences', preferences)
    print("Thank you for sharing. I'll remember that about you.")

def get_user_context(core_memory):
    user_name = core_memory.get_memory('user_name')
    preferences = core_memory.get_memory('preferences')
    context = f"User Context:\nName: {user_name}\n"
    if preferences:
        context += "Preferences:\n"
        for pref in preferences:
            context += f"- {pref['preference']} (shared on {pref['date']})\n"
    return context

def main():
    print("Welcome to the Advanced AI Assistant with Core Memory!")
    print("Type 'load context' to load a previous conversation.")
    print("Type 'save context' to save the current conversation.")
    print("Type 'read pdf <file_path>' to read and analyze a PDF file.")
    print("Type 'update preferences' to add personal information.")
    print("Type 'show my info' to display your stored information.")
    print("Type 'exit' or 'quit' to end the session.")

    core_memory = CoreMemory()

    if not core_memory.get_memory('user_name'):
        initialize_user(core_memory)

    personalize_system_message(system_message, core_memory)

    messages = [system_message]
    current_context = None

    while True:
        question = input("You: ")

        if question.lower() in ['exit', 'quit']:
            if current_context:
                save_conversation_context(messages, current_context)
            break
        elif question.lower() == "save context":
            current_context = save_conversation_context(messages)
            continue
        elif question.lower() == "load context":
            contexts = list_saved_contexts()
            if contexts:
                choice = input("Enter the number of the context to load: ")
                try:
                    index = int(choice) - 1
                    if 0 <= index < len(contexts):
                        messages = load_conversation_context(contexts[index])
                        current_context = contexts[index]
                    else:
                        print("Invalid choice. Please try again.")
                except ValueError:
                    print("Invalid input. Please enter a number.")
            continue
        elif question.lower().startswith("read pdf"):
            file_path = question[9:].strip()
            analysis = handle_pdf_analysis(file_path)
            if analysis:
                messages.append({"role": "user", "content": f"I've read and analyzed a PDF. Here's the analysis:\n\n{analysis}\n\nCan you provide a concise summary of the main points and any key insights from this analysis?"})
                response = get_completion(messages)
                messages.append({"role": "assistant", "content": response})
        elif question.lower() == "update preferences":
            update_user_preferences(core_memory)
            continue
        elif question.lower() == "show my info":
            print(get_user_context(core_memory))
            continue
        else:
            # Add user context to the message for better personalization
            user_context = get_user_context(core_memory)
            messages.append({"role": "system", "content": user_context})

            if should_search_academic(question):
                academic_info = get_academic_info(question)
                messages.append({"role": "system", "content": f"Recent academic information related to the user's query:\n\n{academic_info}"})
            
            messages.append({"role": "user", "content": question})
            response = get_completion(messages)
            messages.append({"role": "assistant", "content": response})

        # Autosave every 5 messages
        if len(messages) % 5 == 0:
            current_context = save_conversation_context(messages, current_context)
if __name__ == '__main__':
    main()