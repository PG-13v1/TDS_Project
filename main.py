import os
import sys
import subprocess

# List of required packages

def install_missing_packages():
    """Ensure all required packages are installed using uv."""
    try:
        REQUIRED_PACKAGES = [
    "uv",
    "fastapi",
    "pydantic",
    "requests",
    "pillow",
    "pytesseract",
    "sentence-transformers",
    "db-sqlite3",
    "pandas",
    "faker",
    "markdown",
    "beautifulsoup4",
    "yake",
    "rapidfuzz",
    "git+https://github.com/openai/whisper.git",
    "easyocr",
    "uvicorn",
    "config",
    "googletrans-py"
]
        
        subprocess.run(["uv", "pip", "install"] + REQUIRED_PACKAGES, check=True)
    except ImportError:
        print("uv is not installed. Please install it first using `pip install uv`.")

print("Installing missing packages...")
install_missing_packages()

import uv
from fastapi import FastAPI, HTTPException, Query
from fastapi import Response,Request
from fastapi.responses import JSONResponse
from fastapi.responses import HTMLResponse
from pydantic import BaseModel
from typing import Optional, Dict, Any
import subprocess
import json
import os
import requests
from pathlib import Path
from datetime import datetime
from PIL import Image,ImageEnhance
import pytesseract
from sentence_transformers import SentenceTransformer, util
import sqlite3
import pandas as pd
import faker
import config
import re
import shutil
import sys
import easyocr
import uv
import uvicorn
import markdown
from bs4 import BeautifulSoup
import yake

import time
import csv
from rapidfuzz import process, fuzz
import whisper
from googletrans import Translator


# Response Models
class BaseResponse(BaseModel):
    success: bool
    message: str
    error: Optional[str] = None
    status: int

class TaskRequest(BaseModel):
    task: str

class TaskResponse(BaseResponse):
    task: Optional[str] = None
    file: Optional[str] = None

class ReadFileResponse(BaseResponse):
    content: Optional[str] = None
    file: Optional[str] = None

app = FastAPI(
    title="Task Automation API",
    description="API for executing various automation tasks and reading files",
    version="1.0.0"
)

AIPROXY_TOKEN = os.environ.get("AIPROXY_TOKEN")
if not AIPROXY_TOKEN:
    raise ValueError("Missing AIPROXY_TOKEN environment variable")

def query_llm(prompt: str) -> str:
    """Queries the LLM to interpret the task or extract information."""
    headers = {
        "Content-Type": "application/json",
        "Authorization": f"Bearer {AIPROXY_TOKEN}"
    }
    
    payload = {
        "model": "gpt-4o-mini",
        "messages": [
            {
                "role": "system", 
                "content": """You are an expert automation assistant. Parse tasks and return JSON with 'action' and 'params',You MUST return ONLY valid JSON without any formatting or explanations.
                Here are the supported actions and their variations:


                0. fetch_api_data:
                    - given the .py script run it and
                    - fetch the url
                    - fetch the email
                    - use this only if datagen.py is mentioned in path url else not
                    - Handle Multimodality 
                    Returns: {"action": "fetch_api_data", "params": {"api_url": "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py", "email": "25ds1000038@ds.study.iitm.ac.in"}} 

                1. retrieve_logs:
                   - Get recent log files
                   - Show latest logs
                   - Recent log entries
                   - Extract from logs
                   - View log files
                   - Show the specific number of files
                   - Handle Multimodality
                   - if in the input path 'data/' or '/data/' is mentioned only take the part after it 
                   Returns: {"action": "retrieve_logs", "params": {"num_logs": <number>, "input": "logs", "output": "logs-recent.txt"}}

                2. format_markdown:
                   - Format markdown file
                   - Prettify markdown
                   - Handle Multimodality
                   - if in the input path 'data/' or '/data/' is mentioned only take the part after it
                   Returns: {"action": "format_markdown", "params": {"input":"format.md"}}

                3. count_weekdays:
                   - Count number of Mondays
                   - Count number of particular days of each one asked as Monday, Tuesday, Wednesday, Thursday, Friday, Saturday, Sunday
                   - if in the input path 'data/' or '/data/' is mentioned only take the part after it
                   - Handle Multimodality
                   Returns: {"action": "count_weekdays", "params": {"weekday": "Monday", "input": "dates.txt", "output": "dates-count.txt"}}

                4. sort_contacts:
                   - Sort contacts by name
                   - Order contacts
                   - if in the input path 'data/' or '/data/' is mentioned only take the part after it
                   - Handle Multimodality
                   Returns: {"action": "sort_contacts", "params": {"input": "", "output": "contacts-sorted.json"}}

                Always map similar phrases to these core actions. When in doubt about the number of items, default to 10.

                5. index_markdown:
                   - Create index of markdown files
                   - Extract markdown titles
                   - Find markdown headings
                   - Create markdown index
                   - Index md files
                   - if in the input path 'data/' or '/data/' is mentioned only take the part after it
                   - Handle Multimodality
                   Returns: {"action": "index_markdown", "params": {"input_dir": "docs", "output_file": "docs/index.json"}}
                
                Always map similar phrases to these core actions. For email extraction tasks, focus on finding the sender's email address.
                Extract only the email address itself, not any surrounding text or formatting.

                6. email_extraction:
                   - Extract email address from email
                   - Find sender's email
                   - if in the input path 'data/' or '/data/' is mentioned only take the part after it
                   - look for field like sender,From,forwrard and extract email from there
                   - the email moght lok like {Regex: (?i)[a-z0-9._%+-]+@[a-z0-9.-]+\\.[a-z]{2,}}
                   - Handle Multimodality
                   Returns: {"action": "email_extraction", "params": {"input": "email.txt", "output": "email-sender.txt"}}

                7. credit_card_number_extraction:
                   - Extract credit card number from text
                   - Find credit card number
                   - Write without spaces
                   - this is an image file
                   - write the credit card number without spaces
                   - if in the input path 'data/' or '/data/' is mentioned only take the part after ith
                   - Handle Multimodality
                   Returns: {"action": "credit_card_number_extraction", "params": {"input": "credit_card.png", "output": "credit-card.txt"}}
                
                8. similar_comments:
                   - Find similar comments
                   - Use cosine similarity
                   - Use embeddings to find similar texts
                   - Use embeddings per line
                   - if in the input path 'data/' or '/data/' is mentioned only take the part after it
                   - Handle Multimodality
                   Returns: {"action": "similar_comments", "params": {"input": "comments.txt", "output": "similar-comments.txt"}}
                
                9. get_tickets_databse_query:
                   - give the text file for a specific query
                   - get the output in the text file
                   - if in the input path 'data/' or '/data/' is mentioned only take the part after it
                   - so i am gonna ask you to give some, data in english convert that into a sqlite query in put it in the query params
                   - the query will be normal for countting average or summin total of a specific column
                   - this function will only be called when talking abou the tickets databse not for any other database
                   - Handle Multimodality
                   Returns: {"action": "get_tickets_database_query", "params": {"input": "ticket-sales.db", "output": "gold-price.txt","query": "SELECT * FROM tickets WHERE type = 'Gold'"}}
                
                10. clone_git_repo:
                   - Clone a git repository
                   - Make a commit
                   - Clone a repo and commit
                   - Handle Multimodality
                   - if in the input path 'data/' or '/data/' is mentioned only take the part after it
                   Returns: {"action": "git_clone_repo", "params": {"repo_url": "","user_name":"","user_email":"", "commit_message": "commiting repository","user_PAT":""}}
                
                11. get_databse_query:
                   - give the text file for a specific query
                   - get the output in the text file
                   - if in the input path 'data/' or '/data/' is mentioned only take the part after it
                   - so i am gonna ask you to give some, data in english convert that into a sqlite query in put it in the query params
                   - this should be used when calling any other database other than tickets
                   - Handle Multimodality
                     Returns: {"action": "get_database_query", "params": {"input": "data.db", "output": "databse_query.txt","query": "SELECT * FROM data"}}
                   
                12. scrape_website:
                  - Scraping a website using beuaitiful soup
                  - Extracting data from a website
                  - Extracting data from a webpage
                  - if in the input path 'data/' or '/data/' is mentioned only take the part after it
                  - Handle Multimodality
                  Returns: {"action": "scrape_website", "params": {"url": "https://www.example.com", "output": "data.json"}}

                13. compress_resize_image:
                  - image would be given
                  - compress the image losslessely if asked to compress it
                  - resize image if asked to resize ,set the compress param to False if not asked to compress
                  - if it says do not compress set compress to False
                  - if in the input path 'data/' or '/data/' is mentioned only take the part after it
                  - Handle Multimodality
                  Returns: {"action": "compress_resize_image", "params": {"input": "input.jpg", "output": "img_output.jpg", "compresss": "True","width": 800, "height": 600}}

                14. transcribe_audio_file
                   - Transcribe an audio file
                   - Convert audio to text
                   - Extract text from audio
                   - if in the input path 'data/' or '/data/' is mentioned only take the part after it
                   - Handle Multimodality
                   Returns: {"action": "transcribe_audio_file", "params": {"input": "sample-1.mp3", "output": "audio.txt"}}
                
                15. markdown_to_html
                   - Convert markdown to html
                   - Convert md to html
                   - Convert markdown to html
                   - Handle Multimodality
                   - if in the input path 'data/' or '/data/' is mentioned only take the part after it
                   Returns: {"action": "markdown_to_html", "params": {"input": "format.md", "output": "sample.html"}}
                
                16. filter_csv_file
                   - user will provide a csv file
                   - filter the csv file based on the given condition
                   - multiple filters may be applied
                   - if in any request '/data/something.txt is mentioned only take the file name and not the full path
                   - if the use mentions greater than use > and for less than use <
                   - if the user mentions equal to use =
                   - if in the input path 'data/' or '/data/' is mentioned only take the part after it
                   - Handle Multimodality
                     Returns: {"action": "filter_csv_file", "params": {"input": "input.csv", "output": "filtered-data.csv","filters": {"column": "", "operator": "", "value": }}}

                17. translate_text
                   - Translate text to another language
                   - Convert text to another language
                   - Translate text to a different language
                   - Handle Multimodality
                   - if in the input path 'data/' or '/data/' is mentioned only take the part after it
                     Returns: {"action": "translate_text", "params": {"input": "sample.txt", "output": "translated.txt","src_language": "fr","target_language":"en"}}
                
                18. fetch_external_api
                   - fetch data if the user mentions to fetch data from an external api
                   - if datagen.py is in the path then run dont use this script
                   Returns: {"action": "fetch_external_api", "params": {"api_url": "https://api.example.com/data", "output": "data.json"}}
                """
            },
            {
                "role": "user", 
                "content": prompt
            }
        ],
        "max_tokens": 150,
        "temperature": 0.3,
        "stream": False
    }
    
    try:
        response = requests.post(
            "http://aiproxy.sanand.workers.dev/openai/v1/chat/completions",
            headers=headers,
            json=payload,
            timeout=30
        )
        
        # Print debug information
        print(f"Request URL: http://aiproxy.sanand.workers.dev/openai/v1/chat/completions")
        print(f"Request Headers: {headers}")
        print(f"Request Payload: {json.dumps(payload, indent=2)}")
        print(f"Response Status: {response.status_code}")
        print(f"Response Body: {response.text}")
        
        if response.status_code != 200:
            raise HTTPException(
                status_code=500, 
                detail=f"LLM request failed with status {response.status_code}: {response.text}"
            )
            
        response_data = response.json()
        if not response_data.get("choices") or not response_data["choices"][0].get("message"):
            raise HTTPException(
                status_code=500,
                detail=f"Invalid response format from LLM: {response.text}"
            )
            
        return response_data["choices"][0]["message"]["content"].strip()
    except requests.exceptions.RequestException as e:
        print(f"Request Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to communicate with LLM service: {str(e)}"
        )
    except json.JSONDecodeError as e:
        print(f"JSON Error: {str(e)}, Response: {response.text}")
        raise HTTPException(
            status_code=500,
            detail=f"Failed to parse LLM response: {response.text}"
        )
    except Exception as e:
        print(f"Unexpected Error: {str(e)}")
        raise HTTPException(
            status_code=500,
            detail=f"Unexpected error: {str(e)}"
        )
    
def safe_path(relative_path: str) -> str:
    """
    Resolve the given relative path against DATA_DIR and verify it is within DATA_DIR.
    Raises a ValueError if the resolved path is outside DATA_DIR.
    """
    # Join the relative path to the base directory
    full_path = os.path.abspath(os.path.join(DATA_DIR, relative_path))
    # Ensure the resolved path starts with DATA_DIR
    if not full_path.startswith(os.path.abspath(DATA_DIR)):
        raise ValueError("Access to files outside the data directory is not allowed.")
    return full_path
    
def execute_command(command: str,directory:str) -> str:
    """Safely execute a shell command."""
    try:
        forbidden_substrings = ["rmdir", "rm -", "del"]
        if any(forbidden in command.lower() for forbidden in forbidden_substrings):
              raise ValueError("Removal commands are not allowed in execute_command.")
        
        result = subprocess.run(
            command,
            shell=True,
            check=True,
            capture_output=True,
            text=True,
            cwd=directory  # Set working directory to DATA_DIR
        )
        return result.stdout.strip()
    except subprocess.CalledProcessError as e:
        raise HTTPException(status_code=500, detail=f"Command failed: {e.stderr}")
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Error executing command: {str(e)}")   

def load_config():
    url = "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py"

    # Fetch the script content
    response = requests.get(url)
    script_content = response.text

    # Extract "config" dictionary using regex
    match = re.search(r'config\s*=\s*(\{.*?\})', script_content, re.DOTALL)
    if match:
        config_str = match.group(1)
        config = eval(config_str)  # Converts string to dictionary (unsafe for untrusted sources)
        root_directory = config.get("root", "data")
        absolute_root_directory = os.path.join(os.getcwd(), root_directory)
        print(f"Extracted root directory: {absolute_root_directory}")
        return absolute_root_directory
    else:
        print("Could not extract 'config' dictionary.")
        return os.path.join(os.getcwd(), "data")

def handle_fetch_api_data(params:Dict) -> dict:
                        
    DATA_DIR = load_config()
    print(f'CWD: {os.getcwd()}')
    print(f'DATA_DIR: {DATA_DIR}')
    os.makedirs(DATA_DIR, exist_ok=True)

    api_url = params.get("api_url", "https://raw.githubusercontent.com/sanand0/tools-in-data-science-public/tds-2025-01/project-1/datagen.py")
    email = params.get("email", "25ds1000038@ds.study.iitm.ac.in")

    print(f"Fetching data from API: {api_url}")


    script_path = os.path.join(DATA_DIR, "datagen.py")

    try:
     # Download script
     response = requests.get(api_url, timeout=10)
     response.raise_for_status()
    
     with open(script_path, "w", encoding="utf-8") as f:
        f.write(response.text)
    
     print(f"Script saved at {script_path}")

     # Run the script using uv
     output = execute_command(f"uv run {script_path} {email}", DATA_DIR)

    # Move the data folder if necessary
     if os.getcwd() != DATA_DIR:
        destination_dir = os.getcwd()
        try:
            shutil.move(DATA_DIR, destination_dir)
            print(f"Moved {DATA_DIR} to {destination_dir}")
        except Exception as e:
            print(f"Error moving directory: {str(e)}")

     return {
        "status_code": 200,
        "response": {
            "success": True,
            "message": "Python script executed successfully",
            "output": output,
            "status": 200
        }}
    except requests.exceptions.RequestException as e:
       print(f"Error downloading script: {str(e)}")

    return {
        "status_code": 500,
        "response": {
            "success": False,
            "message": "Error downloading script",
            "error": str(e),
            "status": 500
        }
    }

DATA_DIR =os.getcwd()+'/data'

def handle_format_markdown(params: Dict) -> dict:
    try:
        # Install prettier and its markdown plugin
        execute_command("npm install -g prettier@3.4.2 @prettier/plugin-markdown", DATA_DIR)
        
        # Ensure data directory exists
        os.makedirs(DATA_DIR, exist_ok=True)

        file = params.get("input", "format.md")  # Change to match API spec
        
        markdown_file = os.path.join(DATA_DIR, file)
        
        # Create empty markdown file if it doesn't exist
        if not os.path.exists(markdown_file):
            with open(markdown_file, 'w') as f:
                f.write('# Empty Markdown\n')
        
        if not Path(markdown_file).is_file():
            return {
                "status_code": 404,
                "response": {
                    "success": False,
                    "message": "Markdown file not found",
                    "error": "File does not exist",
                    "status": 404
                }
            }
            
        # Use prettier with markdown plugin
        execute_command(f"npx prettier --parser markdown --write {markdown_file}", DATA_DIR)
        
        return {
            "status_code": 200,
            "response": {
                "success": True,
                "message": "Markdown formatted successfully",
                "task": "format_markdown",
                "file": file,  # Return the actual file name used
                "status": 200
            }
        }
    except Exception as e:
        return {
            "status_code": 500,
            "response": {
                "success": False,
                "message": "Error formatting markdown",
                "error": str(e),
                "status": 500
            }
        }

def handle_count_weekdays(params: Dict) -> dict:
    """Handle counting weekdays in a dates file."""
    try: 
        # Get parameters with defaults
        input_file = params.get("input", "dates.txt")
        output_file = params.get("output", "dates-wednesdays.txt")
        weekday_name = params.get("weekday", "Wednesday")
        
        # Map weekday names to numbers (0 = Monday, 6 = Sunday)
        weekday_map = {
            "monday": 0, "tuesday": 1, "wednesday": 2, "thursday": 3,
            "friday": 4, "saturday": 5, "sunday": 6
        }
        weekday_num = weekday_map[weekday_name.lower()]
        
        # Read dates and count weekdays
        input_path = os.path.join(DATA_DIR, input_file.replace('/', os.sep).replace('\\', os.sep))
        output_path = os.path.join(DATA_DIR, output_file.replace('/', os.sep).replace('\\', os.sep))
        
        # Create the data directory if it doesn't exist
        os.makedirs(DATA_DIR, exist_ok=True)
        
        print(f"Writing to path: {output_path}")  # Debug print
        print(f"Directory exists: {os.path.exists(os.path.dirname(output_path))}")  # Debug print
        print(f"Directory is writable: {os.access(os.path.dirname(output_path), os.W_OK)}")  # Debug print
        
        if not os.path.exists(input_path):
            return {
                "status_code": 404,
                "response": {
                    "success": False,
                    "message": f"Input file not found: {input_file}",
                    "error": "File does not exist",
                    "status": 404
                }
            }
        try:
            with open(input_path, 'r') as f:
                dates = f.readlines()
            count = 0
            for date_str in dates:
                date_str = date_str.strip()
                try:
                    # Try YYYY-MM-DD format
                    date = datetime.strptime(date_str, "%Y-%m-%d")
                except ValueError:
                    try:
                        # Try DD-MMM-YYYY format
                        date = datetime.strptime(date_str, "%d-%b-%Y")
                    except ValueError:
                        try:
                            # Try MMM DD, YYYY format
                            date = datetime.strptime(date_str, "%b %d, %Y")
                        except ValueError:
                            try:
                                # Try YYYY/MM/DD HH:MM:SS format
                                date = datetime.strptime(date_str, "%Y/%m/%d %H:%M:%S")
                            except ValueError:
                                return {
                                    "status_code": 400,
                                    "response": {
                                        "success": False,
                                        "message": "Invalid date format in input file",
                                        "error": f"Date '{date_str}' is not in a recognized format (YYYY-MM-DD, DD-MMM-YYYY, MMM DD, YYYY, or YYYY/MM/DD HH:MM:SS)",
                                        "status": 400
                                    }
                                }
                
                if date.weekday() == weekday_num:
                    count += 1
            
            # Write result to output file
            try:
                with open(output_path, 'w') as f:
                    f.write(str(count))
                print(f"File written successfully. Content: {count}")  # Debug print
                print(f"File exists after writing: {os.path.exists(output_path)}")  # Debug print
                if os.path.exists(output_path):
                    with open(output_path, 'r') as f:
                        print(f"File content after writing: {f.read()}")  # Debug print
            except Exception as write_error:
                print(f"Error writing to file: {str(write_error)}")  # Debug print
                raise
            
            return {
                "status_code": 200,
                "response": {
                    "success": True,
                    "message": f"Counted {count} {weekday_name}s",
                    "task": "count_weekdays",
                    "file": output_file,
                    "count": count,
                    "status": 200
                }
            }
            
        except Exception as e:
            return {
                "status_code": 400,
                "response": {
                    "success": False,
                    "message": "Error processing dates",
                    "error": str(e),
                    "status": 400
                }
            }
            
    except Exception as e:
        return {
            "status_code": 500,
            "response": {
                "success": False,
                "message": "Error counting weekdays",
                "error": str(e),
                "status": 500
            }
        }

def handle_contact_name(params: Dict) -> dict:
        """Handle sorting contacts by name."""
        try:
            input_file = params.get("input", "contacts.json")
            output_file = params.get("output", "contacts-sorted.json")
        
            input_file = os.path.join(DATA_DIR, input_file)
            output_file = os.path.join(DATA_DIR, output_file)

            if not os.path.exists(input_file):
                return {
                    "status_code": 404,
                    "response": {
                        "success": False,
                        "message": "Contacts file not found",
                        "error": "File does not exist",
                        "status": 404
                    }
                }

            with open(input_file, 'r') as f:
                contacts = json.load(f)

            sorted_contacts = sorted(contacts, key=lambda x: (x['last_name'], x['first_name']))

            with open(output_file, 'w') as f:
                json.dump(sorted_contacts, f)

            with open(output_file, 'r') as f:
                sorted_contacts_content = f.read()
            
            print(f"Sorted contacts content: {sorted_contacts_content}")

            return {
                "status_code": 200,
                "response": {
                    "success": True,
                    "message": "Contacts sorted successfully",
                    "task": "sorted_contacts_by_name",
                    "file": "contacts-sorted.json",
                    "status": 200
                }
            }

        except Exception as e:
            return {
                "status_code": 500,
                "response": {
                    "success": False,
                    "message": "Error sorting contacts",
                    "error": str(e),
                    "status": 500
                }
            }
        
def handle_recent_log_file(params: Dict) -> dict:
            """Handle extracting first lines from recent log files."""
            try:
                num_logs = int(params.get("num_logs", 10))
                input_dir = params.get("input", "logs")
                output_file = params.get("output", "logs-recent.txt")
                
                # Set default paths
                if os.path.isabs(input_dir):
                    logs_dir = input_dir
                else:
                    logs_dir = os.path.join(DATA_DIR, input_dir)

                if os.path.isabs(output_file):
                    output_path = output_file  
                else:
                    output_path = os.path.join(DATA_DIR, output_file)
                # Create directories if they don't exist
                os.makedirs(logs_dir, exist_ok=True)
                

                if not os.path.exists(logs_dir):
                    return {
                        "status_code": 404,
                        "response": {
                            "success": False,
                            "message": "Logs directory not found",
                            "error": "Directory does not exist",
                            "status": 404
                        }
                    }

                # Get all .log files and sort by modification time
                log_files = []
                for file in os.listdir(logs_dir):
                    if file.endswith('.log'):
                        full_path = os.path.join(logs_dir, file)
                        log_files.append((full_path, os.path.getmtime(full_path)))
                
                log_files.sort(key=lambda x: x[1], reverse=True)
                log_files = log_files[:num_logs]

                # Extract first lines
                first_lines = []
                for file_path, _ in log_files:
                    try:
                        with open(file_path, 'r') as f:
                            first_line = f.readline().strip()
                            file_name = os.path.basename(file_path)
                            first_lines.append(first_line)
                    except Exception as e:
                        print(f"Error reading file {file_path}: {str(e)}")
                        continue

                # Write results to output file
                with open(output_path, 'w') as f:
                    f.write('\n'.join(first_lines))

                return {
                    "status_code": 200,
                    "response": {
                        "success": True,
                        "message": f"Extracted first lines from {len(first_lines)} log files",
                        "task": "recent_log_file",
                        "file": output_file,
                        "status": 200
                    }
                }

            except Exception as e:
                return {
                    "status_code": 500,
                    "response": {
                        "success": False,
                        "message": "Error processing log files",
                        "error": str(e),
                        "status": 500
                    }
                }

def index_markdown(params: Dict) -> dict:
    try:
        input_dir = params.get("input_dir", "docs")
        output_file = params.get("output_file", "docs/index.json")

        # Build full paths relative to DATA_DIR
        input_dir = os.path.join(DATA_DIR, input_dir)
        output_file = os.path.join(DATA_DIR, output_file)

        if not os.path.exists(input_dir):
            return {
                "status_code": 404,
                "response": {
                    "success": False,
                    "message": "Input directory not found",
                    "error": "Directory does not exist",
                    "status": 404
                }
            }

        index = {}
        # Iterate over all markdown files in input_dir
        for root, _, files in os.walk(input_dir):
            for file in files:
                if file.endswith(".md"):
                    file_path = os.path.join(root, file)
                    # Read the file contents to generate a summary (here, first 50 characters)
                    with open(file_path, 'r', encoding='utf-8') as f:
                        content = f.read().strip()
                        summary = content[:50] if content else ""
                    # Use only the file name as key (e.g. language.md)
                    index[file] = summary

        # Ensure the directory for the output file exists
        os.makedirs(os.path.dirname(output_file), exist_ok=True)
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(index, f, indent=2)

        return {
            "status_code": 200,
            "response": {
                "success": True,
                "message": "Markdown index created successfully",
                "task": "index_markdown",
                "file": os.path.basename(output_file),
                "status": 200
            }
        }

    except Exception as e:
        return {
            "status_code": 500,
            "response": {
                "success": False,
                "message": "Error creating markdown index",
                "error": str(e),
                "status": 500
            }
        }

def handle_email_extraction(params: Dict) -> dict:
    """Handle extracting the sender's email address from an email file."""
    try:
        input_file = params.get("input", "email.txt")
        output_file = params.get("output", "email-sender.txt")

        input_path = os.path.join(DATA_DIR, input_file)
        output_path = os.path.join(DATA_DIR, output_file)

        if not os.path.exists(input_path):
            return {
                "status_code": 404,
                "response": {
                    "success": False,
                    "message": "Email file not found",
                    "error": "File does not exist",
                    "status": 404
                }
            }

        with open(input_path, 'r') as f:
            email_content = f.read()

        # First try to find email in From: field
        from_match = re.search(r'From:.*?([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', email_content, re.IGNORECASE | re.MULTILINE)
        
        if from_match:
            email_address = from_match.group(1)
        else:
            # Fallback to finding any email address
            email_match = re.search(r'([a-zA-Z0-9._%+-]+@[a-zA-Z0-9.-]+\.[a-zA-Z]{2,})', email_content)
            if email_match:
                email_address = email_match.group(1)
            else:
                return {
                    "status_code": 400,
                    "response": {
                        "success": False,
                        "message": "No email address found",
                        "error": "Could not find email address in content",
                        "status": 400
                    }
                }

        with open(output_path, 'w') as f:
            f.write(email_address)

        return {
            "status_code": 200,
            "response": {
                "success": True,
                "message": "Email address extracted successfully",
                "task": "email_extraction",
                "e-mail": email_address,
                "file": output_file,
                "status": 200
            }
        }

    except Exception as e:
        return {
            "status_code": 500,
            "response": {
                "success": False,
                "message": "Error extracting email address",
                "error": str(e),
                "status": 500
            }
        }

def handle_credit_card_number_extraction(params: Dict) -> dict:
    """Handle extracting the credit card number from an image file."""
    try:
        input_file = params.get("input", "credit-card.png")
        output_file = params.get("output", "credit-card.txt")

        input_path = os.path.join(DATA_DIR, input_file)
        output_path = os.path.join(DATA_DIR, output_file)

        if not os.path.exists(input_path):
            return {
                "status_code": 404,
                "response": {
                    "success": False,
                    "message": "Image file not found",
                    "error": "File does not exist",
                    "status": 404
                }
            }

        # Use pytesseract to extract text from the image
        image = Image.open(input_path)
        image = image.convert('L')
        enhancer = ImageEnhance.Contrast(image)
        image = enhancer.enhance(2.0)

        temp_path = os.path.join(DATA_DIR, "temp_processed.png")
        image.save(temp_path)

        reader = easyocr.Reader(['en'],gpu=False, model_storage_directory=None)
        
        # Read text from image
        results = reader.readtext(temp_path, paragraph=False, 
                                decoder='greedy',
                                beamWidth=5,
                                batch_size=1,
                                workers=0,
                                allowlist='0123456789- ')

        # Clean up temp file
        os.remove(temp_path)
        extracted_text = " ".join([text for (bbox, text, prob) in results])
        print(f"Extracted text: {extracted_text}")

        # Query the LLM to extract the credit card number
        prompt = f"Extract the credit card number from the following text:\n\n{extracted_text}"
        structured_task = query_llm(prompt)

        patterns = [
            r'\b\d{16}\b',                                    # Basic 16 digits
            r'\b\d{4}[\s-]?\d{4}[\s-]?\d{4}[\s-]?\d{4}\b',  # With optional separators
            r'\b\d{4}\s?\d{6}\s?\d{5}\b',                    # Alternative format
            r'[0-9]{4}[0-9\s-]{8,12}[0-9]{4}'               # Flexible pattern
        ]

        card_number = None
        for pattern in patterns:
            matches = re.findall(pattern, extracted_text)
            if matches:
                # Clean up the matched number
                card_number = re.sub(r'[\s-]', '', matches[0])
                if len(card_number) >= 15:  # Valid card numbers are 15-16 digits
                    break
        
        if not card_number:
            return {
                "status_code": 400,
                "response": {
                    "success": False,
                    "message": "No credit card number found",
                    "error": "Could not detect credit card number in image",
                    "status": 400
                }
            }

        with open(output_path, 'w') as f:
            f.write(card_number)

        return {
            "status_code": 200,
            "response": {
                "success": True,
                "message": "Credit card number extracted successfully",
                "task": "credit_card_number_extraction",
                "file": output_file,
                "status": 200
            }
        }

    except Exception as e:
        print(f"Error: {str(e)}")  # Debug print
        return {
            "status_code": 500,
            "response": {
                "success": False,
                "message": "Error extracting credit card number",
                "error": str(e),
                "status": 500
            }
        }
  
def handle_similar_comments(params: Dict) -> dict:
    """Handle finding the most similar pair of comments using embeddings."""
    try:
        input_file = params.get("input", "comments.txt")
        output_file = params.get("output", "comments-similar.txt")

        input_path = os.path.join(DATA_DIR, input_file)
        output_path = os.path.join(DATA_DIR, output_file)

        if not os.path.exists(input_path):
            return {
                "status_code": 404,
                "response": {
                    "success": False,
                    "message": "Comments file not found",
                    "error": "File does not exist",
                    "status": 404
                }
            }

        with open(input_path, 'r') as f:
            comments = f.readlines()

        if len(comments) < 2:
            return {
                "status_code": 400,
                "response": {
                    "success": False,
                    "message": "Not enough comments to find a similar pair",
                    "error": "File must contain at least two comments",
                    "status": 400
                }
            }

        # Load the pre-trained model
        model = SentenceTransformer('all-MiniLM-L6-v2')

        # Compute embeddings for all comments
        embeddings = model.encode(comments, convert_to_tensor=True)

        # Compute cosine similarities
        cosine_scores = util.pytorch_cos_sim(embeddings, embeddings)

        # Find the pair with the highest similarity score
        max_score = -1
        most_similar_pair = (None, None)
        for i in range(len(comments)):
            for j in range(i + 1, len(comments)):
                if cosine_scores[i][j] > max_score:
                    max_score = cosine_scores[i][j]
                    most_similar_pair = (comments[i].strip(), comments[j].strip())

        if most_similar_pair == (None, None):
            return {
                "status_code": 400,
                "response": {
                    "success": False,
                    "message": "No similar comments found",
                    "error": "Could not find a pair of similar comments",
                    "status": 400
                }
            }

        with open(output_path, 'w') as f:
            f.write(most_similar_pair[0] + '\n')
            f.write(most_similar_pair[1] + '\n')

        return {
            "status_code": 200,
            "response": {
                "success": True,
                "message": "Most similar comments found successfully",
                "task": "similar_comments",
                "file": output_file,
                "status": 200
            }
        }

    except Exception as e:
        return {
            "status_code": 500,
            "response": {
                "success": False,
                "message": "Error finding similar comments",
                "error": str(e),
                "status": 500
            }
        }

def handle_database_query(params: Dict) -> dict:
    """Handle calculating the total sales of 'Gold' ticket type."""
    try:
        db_file = params.get("input", "ticket-sales.db")
        output_file = params.get("output", "ticket-sales-gold.txt")
        query=params.get("query","SELECT * FROM tickets WHERE type = 'Gold'")

        db_path = os.path.join(DATA_DIR, db_file)
        output_path = os.path.join(DATA_DIR, output_file)

        if not os.path.exists(db_path):
            return {
                "status_code": 404,
                "response": {
                    "success": False,
                    "message": "Database file not found",
                    "error": "File does not exist",
                    "status": 404
                }
            }

        # Connect to the SQLite database
        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()

        # Query to calculate the total sales of 'Gold' ticket type
        cursor.execute(query)
        total_sales = cursor.fetchone()[0]

        # Close the database connection
        conn.close()

        if total_sales is None:
            total_sales = 0

        with open(output_path, 'w') as f:
            f.write(str(total_sales))

        return {
            "status_code": 200,
            "response": {
                "success": True,
                "message": "Total sales of 'Gold' ticket type calculated successfully",
                "task": "gold_type_price",
                "file": output_file,
                "status": 200
            }
        }

    except Exception as e:
        return {
            "status_code": 500,
            "response": {
                "success": False,
                "message": "Error calculating total sales of 'Gold' ticket type",
                "error": str(e),
                "status": 500
            }
        }

def handle_clone_git_repo(params: Dict) -> dict:
    """Handle cloning a git repo and making a commit."""
    try:
        repo_url = params.get("repo_url", None)
        commit_message = params.get("commit_message", "Automated commit")
        user_name = params.get("user_name", "PG-13v1")
        user_email = params.get("user_email", "working.pratul@gmail.com")
        user_PAT = params.get("user_PAT", "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")

        if not repo_url:
            return {
                "status_code": 400,
                "response": {
                    "success": False,
                    "message": "Repository URL not provided",
                    "error": "Missing repo_url parameter",
                    "status": 400
                }
            }

        # Clean the repo name to be safe for filesystem
        repo_name = os.path.basename(repo_url.rstrip('/')).replace('.git', '')
        repo_path = os.path.join(DATA_DIR, repo_name)

        # Remove directory if it already exists
        if os.path.exists(repo_path):
            # Use proper Windows commands to remove directory with git files
            execute_command(f'rmdir /S /Q "{repo_path}"', DATA_DIR)

        try:
            # Create directory with full permissions
            os.makedirs(repo_path, mode=0o777, exist_ok=True)

            # Set git global config first
            execute_command(f'git config --global user.email "{user_email}"', DATA_DIR)
            execute_command(f'git config --global user.name "{user_name}"', DATA_DIR)

            print(f"Cloning repo from: {repo_url}")

            # Clone using HTTPS with credentials in URL
            auth_url = f'https://{user_name}:{user_PAT}@github.com/{user_name}/{repo_name}.git'
            clone_command = f'git clone "{auth_url}" "{repo_path}"'
            execute_command(clone_command, DATA_DIR)
            
            print(f"Cloned repo to: {repo_path}")

            time.sleep(5)
            if os.path.exists(repo_path):
                print("Repo cloned successfully")
            else:
                print("Repo not cloned")

            # Make commit
            commit_commands = [
    'git add .',
    f'git commit -m "{commit_message}" --allow-empty',
    'git push']
            

            execute_command(" && ".join(commit_commands), repo_path)

            return {
                "status_code": 200,
                "response": {
                    "success": True,
                    "message": "Repository cloned and commit made successfully",
                    "task": "clone_git_repo",
                    "file": repo_name,
                    "status": 200
                }
            }

        except Exception as e:
            # Clean up if something goes wrong
            if os.path.exists(repo_path):
                execute_command(f'rmdir /S /Q "{repo_path}"', DATA_DIR)
            raise e

    except Exception as e:
        return {
            "status_code": 500,
            "response": {
                "success": False,
                "message": "Error cloning git repo or making commit",
                "error": str(e),
                "status": 500
            }
        }

def handle_database_query(params: Dict) -> dict:
    """Handle running a SQL query on a SQLite or DuckDB database."""
    try:
        db_file = params.get("input", "data.db")
        query = params.get("query")
        output_file = params.get("output", "query-result.json")

        if not query:
            return {
                "status_code": 400,
                "response": {
                    "success": False,
                    "message": "SQL query not provided",
                    "error": "Missing query parameter",
                    "status": 400
                }
            }

        db_path = os.path.join(DATA_DIR, db_file)
        output_path = os.path.join(DATA_DIR, output_file)

        if not os.path.exists(db_path):
            return {
                "status_code": 404,
                "response": {
                    "success": False,
                    "message": "Database file not found",
                    "error": "File does not exist",
                    "status": 404
                }
            }

        conn = sqlite3.connect(db_path)
        cursor = conn.cursor()
        cursor.execute(query)
        rows = cursor.fetchall()
        conn.close()

        with open(output_path, 'w') as f:
            json.dump(rows, f, indent=2)

        return {
            "status_code": 200,
            "response": {
                "success": True,
                "message": "SQL query executed successfully",
                "task": "run_sql_query",
                "file": output_file,
                "status": 200
            }
        }

    except Exception as e:
        return {
            "status_code": 500,
            "response": {
                "success": False,
                "message": "Error running SQL query",
                "error": str(e),
                "status": 500
            }
        }

def handle_scrape_website(params: Dict) -> dict:
    """Handle extracting data from a website."""
    try:
        url = params.get("url")
        output_file = params.get("output", "scraped-data.html")

        if not url:
            return {
                "status_code": 400,
                "response": {
                    "success": False,
                    "message": "URL not provided",
                    "error": "Missing url parameter",
                    "status": 400
                }
            }

        response = requests.get(url)
        soup = BeautifulSoup(response.content, 'html.parser')
        page_title = soup.title.text.strip() if soup.title else ""
        # Get all headings (h1 to h6)
        headings = []
        for level in range(1, 7):
            headings.extend([h.get_text(strip=True) for h in soup.find_all(f'h{level}')])
        
        raw_paragraphs = [p.get_text(strip=True) for p in soup.find_all('p')]


        extractor = yake.KeywordExtractor(top=5, stopwords=None)
        paragraphs = []
        for para in raw_paragraphs:
            keywords = extractor.extract_keywords(para)
            # Sort keywords by their score (lower score is more relevant)
            sorted_keywords = [kw for kw, score in sorted(keywords, key=lambda x: x[1])]
            paragraphs.append({
                "text": para,
                "keywords": sorted_keywords
            })

        scraped_data = {
            "title": page_title,
            "headings": headings,
            "paragraphs": paragraphs
        }


        if response.status_code != 200:
            return {
                "status_code": response.status_code,
                "response": {
                    "success": False,
                    "message": "Failed to scrape website",
                    "error": response.text,
                    "status": response.status_code
                }
            }

        file_path=os.path.join(DATA_DIR, output_file)

        with open(file_path, 'w') as f:
            json.dump(scraped_data, f, indent=2)

        return {
            "status_code": 200,
            "response": {
                "success": True,
                "message": "Website scraped and data saved successfully",
                "task": "scrape_website",
                "file": output_file,
                "status": 200
            }
        }

    except Exception as e:
        return {
            "status_code": 500,
            "response": {
                "success": False,
                "message": "Error scraping website",
                "error": str(e),
                "status": 500
            }
        }

def handle_compress_resize_image(params: Dict) -> dict:
    """Handle compressing or resizing an image."""
    try:
        input_file = params.get("input", "image.png")
        output_file = params.get("output", "image-compressed.png")
        width = params.get("width")
        height = params.get("height")
        compress=params.get("compress","True")

        input_path = os.path.join(DATA_DIR, input_file)
        output_path = os.path.join(DATA_DIR, output_file)

        if not os.path.exists(input_path):
            return {
                "status_code": 404,
                "response": {
                    "success": False,
                    "message": "Image file not found",
                    "error": "File does not exist",
                    "status": 404
                }
            }

        image = Image.open(input_path)

        if compress == "False":
            image.save(output_path, optimize=True, quality=90)

        else:
          if width and height:
            image = image.resize((int(width), int(height)))
            image.save(output_path, optimize=True, quality=100)

        return {
            "status_code": 200,
            "response": {
                "success": True,
                "message": "Image compressed/resized successfully",
                "task": "compress_resize_image",
                "file": output_file,
                "status": 200
            }
        }

    except Exception as e:
        return {
            "status_code": 500,
            "response": {
                "success": False,
                "message": "Error compressing/resizing image",
                "error": str(e),
                "status": 500
            }
        }

def handle_transcribe_audio(params: Dict) -> dict:
    """Handle transcribing audio from an MP3 file using OpenAI Whisper."""
    try:
        import whisper

        input_file = params.get("input", "sample-1.mp3")
        # Output a text file with the transcription
        output_file = params.get("output", "sample-1.txt")

        input_path = os.path.join(DATA_DIR, input_file)
        output_path = os.path.join(DATA_DIR, output_file)

        if not os.path.exists(input_path):
            return {
                "status_code": 404,
                "response": {
                    "success": False,
                    "message": "Audio file not found",
                    "error": "File does not exist",
                    "status": 404
                }
            }

        # Load the Whisper model (using the base model for speed, adjust as needed)
        model = whisper.load_model("base")
        # Transcribe the audio file
        result = model.transcribe(input_path)
        transcription = result.get("text", "")

        with open(output_path, 'w') as f:
            f.write(transcription)

        return {
            "status_code": 200,
            "response": {
                "success": True,
                "message": "Audio transcribed successfully",
                "task": "transcribe_audio",
                "file": output_file,
                "status": 200
            }
        }

    except Exception as e:
        return {
            "status_code": 500,
            "response": {
                "success": False,
                "message": "Error transcribing audio",
                "error": str(e),
                "status": 500
            }
        }

def handle_convert_markdown_to_html(params: Dict) -> dict:
    """Handle converting Markdown to HTML."""
    try:
        input_file = params.get("input", "document.md")
        output_file = params.get("output", "document.html")

        input_path = os.path.join(DATA_DIR, input_file)
        output_path = os.path.join(DATA_DIR, output_file)

        if not os.path.exists(input_path):
            return {
                "status_code": 404,
                "response": {
                    "success": False,
                    "message": "Markdown file not found",
                    "error": "File does not exist",
                    "status": 404
                }
            }

        with open(input_path, 'r') as f:
            markdown_content = f.read()

        html_content = markdown.markdown(markdown_content)

        with open(output_path, 'w') as f:
            f.write(html_content)

        return {
            "status_code": 200,
            "response": {
                "success": True,
                "message": "Markdown converted to HTML successfully",
                "task": "convert_markdown_to_html",
                "file": output_file,
                "status": 200
            }
        }

    except Exception as e:
        return {
            "status_code": 500,
            "response": {
                "success": False,
                "message": "Error converting Markdown to HTML",
                "error": str(e),
                "status": 500
            }
        }

def handle_filter_csv(params: Dict) -> dict:
    try:
        input_file = params.get("input", "data.csv")
        output_file = params.get("output", "filtered-data.csv")
        
        filters = params.get("filters", {})
        filter_column = filters.get("column")
        filter_value = filters.get("value")
        filter_operator = filters.get("operator", "=")

        if not filter_column or filter_value is None:
            return {
                "status_code": 400,
                "response": {
                    "success": False,
                    "message": "Filter column or value not provided",
                    "error": "Missing filter_column or filter_value parameter",
                    "status": 400
                }
            }
        
        input_path = os.path.join(DATA_DIR, input_file)
        output_path = os.path.join(DATA_DIR, output_file)

        if not os.path.exists(input_path):
            return {
                "status_code": 404,
                "response": {
                    "success": False,
                    "message": "CSV file not found",
                    "error": "File does not exist",
                    "status": 404
                }
            }

        df = pd.read_csv(input_path)

        # Fuzzy matching if the column name doesn't match exactly
        if filter_column not in df.columns:
            try:
                from rapidfuzz import process, fuzz
            except ImportError:
                return {
                    "status_code": 500,
                    "response": {
                        "success": False,
                        "message": "RapidFuzz library not installed",
                        "error": "Missing rapidfuzz dependency",
                        "status": 500
                    }
                }
            close_match = process.extractOne(filter_column, df.columns, scorer=fuzz.ratio, score_cutoff=80)
            if close_match:
                original = filter_column
                filter_column = close_match[0]
                print(f"Column '{original}' not found. Using closest match '{filter_column}'.")
            else:
                return {
                    "status_code": 400,
                    "response": {
                        "success": False,
                        "message": f"Filter column '{filter_column}' not found and no similar column available.",
                        "error": "Invalid filter_column",
                        "status": 400
                    }
                }

        if filter_operator == "==":
            filtered_df = df[df[filter_column] == filter_value]
        elif filter_operator == ">":
            filtered_df = df[pd.to_numeric(df[filter_column], errors='coerce') > float(filter_value)]
        elif filter_operator == "<":
            filtered_df = df[pd.to_numeric(df[filter_column], errors='coerce') < float(filter_value)]
        else:
            return {
                "status_code": 400,
                "response": {
                    "success": False,
                    "message": f"Unsupported operator '{filter_operator}'",
                    "error": "Invalid operator",
                    "status": 400
                }
            }

        filtered_df.to_csv(output_path, index=False)

        return {
            "status_code": 200,
            "response": {
                "success": True,
                "message": "CSV file filtered and data saved successfully",
                "task": "filter_csv_file",
                "file": output_file,
                "status": 200
            }
        }
    except Exception as e:
        return {
            "status_code": 500,
            "response": {
                "success": False,
                "message": "Error filtering CSV file",
                "error": str(e),
                "status": 500
            }
        }
    
def handle_external_api_request(params: Dict) -> dict:
    """Handle making a request to an external API."""
    try:
        url = params.get("url")
        output_file = params.get("output", "api-response.json")

        if not url:
            return {
                "status_code": 400,
                "response": {
                    "success": False,
                    "message": "API URL not provided",
                    "error": "Missing url parameter",
                    "status": 400
                }
            }

        response = requests.get(url)
        data = response.json()

        file_path = os.path.join(DATA_DIR, output_file)
        with open(file_path, 'w') as f:
            json.dump(data, f, indent=2)

        return {
            "status_code": 200,
            "response": {
                "success": True,
                "message": "API request successful",
                "task": "external_api_request",
                "file": output_file,
                "status": 200
            }
        }

    except Exception as e:
        return {
            "status_code": 500,
            "response": {
                "success": False,
                "message": "Error making API request",
                "error": str(e),
                "status": 500
            }
        }

def handle_translate_text(params: Dict) -> dict:
    """Handle translating a text file using googletrans."""
    try:
        input_file = params.get("input", "sample.txt")
        output_file = params.get("output", "translated.txt")
        src_language = params.get("src_language", None)  # None enables auto-detection
        target_language = params.get("target_language", "en")

        input_path = os.path.join(DATA_DIR, input_file)
        output_path = os.path.join(DATA_DIR, output_file)

        if not os.path.exists(input_path):
            return {
                "status_code": 404,
                "response": {
                    "success": False,
                    "message": "Input text file not found",
                    "error": "File does not exist",
                    "status": 404
                }
            }

        # Read the content from the input file
        with open(input_path, "r", encoding="utf-8") as f:
            text = f.read()

        translator = Translator()
        # Translate the text. If src_language is None, it will auto-detect.
        translation = translator.translate(text, src=src_language, dest=target_language)
        translated_text = translation.text

        # Write the translated text to the output file
        with open(output_path, "w", encoding="utf-8") as f:
            f.write(translated_text)

        return {
            "status_code": 200,
            "response": {
                "success": True,
                "message": "Text translated successfully",
                "task": "translate_text",
                "file": output_file,
                "status": 200
            }
        }

    except Exception as e:
        return {
            "status_code": 500,
            "response": {
                "success": False,
                "message": "Error translating text",
                "error": str(e),
                "status": 500
            }
        }

@app.get("/", response_class=HTMLResponse)
async def get_form():
    return """
    <form action="/run" method="post">
        <input type="text" name="task" placeholder="Enter task">
        <button type="submit">Run Task</button>
    </form>
    """

@app.api_route("/run", 
    methods=["POST", "GET"],  # Allow both POST and GET
    response_model=TaskResponse,
    responses={
        200: {
            "description": "Task executed successfully",
            "model": TaskResponse,
            "content": {
                "application/json": {
                    "example": {
                        "success": True,
                        "message": "Markdown formatted successfully",
                        "task": "format_markdown",
                        "file": "format.md",
                        "status": 200
                    }
                }
            }
        },
        400: {
            "description": "Bad request",
            "model": BaseResponse,
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "message": "Invalid task format",
                        "error": "Could not parse task JSON",
                        "status": 400
                    }
                }
            }
        },
        500: {
            "description": "Internal server error",
            "model": BaseResponse,
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "message": "Internal server error",
                        "error": "Error details",
                        "status": 500
                    }
                }
            }
        }
    }
)
async def run_task(
    request: Request,
    task: str = Query(None, description="Task description from query parameter")
):
    try:
        # If it's a GET request, use the query parameter
        if request.method == "GET":
            if not task:
                raise ValueError("Task parameter is required")
        # If it's a POST request, use the query parameter as well
        else:
            if not task:
                raise ValueError("Task parameter is required")

        print(f"Received task: {task}")
        structured_task = query_llm(f"Parse this task and return a JSON with 'action' and 'params': {task}")
        print(f"Structured task: {structured_task}")
        task_info = json.loads(structured_task)
        print(f"Task info: {task_info}")
        action = task_info.get("action", "").lower()
        print(f"Action: {action}")

        if action == "fetch_api_data":
            result = handle_fetch_api_data(task_info.get("params", {}))
            return JSONResponse(
                status_code=result["status_code"],
                content=result["response"]
            )


        # Handle different actions
        elif action == "format_markdown":
            result = handle_format_markdown(task_info.get("params", {}))
            return JSONResponse(
                status_code=result["status_code"],
                content=result["response"]
            )
        
        elif action == "count_weekdays":
            result = handle_count_weekdays(task_info.get("params", {}))
            return JSONResponse(
                status_code=result["status_code"],
                content=result["response"]
            )
        
        elif action == "sort_contacts":
            result = handle_contact_name(task_info.get("params", {}))
            return JSONResponse(
                status_code=result["status_code"],
                content=result["response"]
            )
        
        elif action == "retrieve_logs":
            result = handle_recent_log_file(task_info.get("params", {}))
            return JSONResponse(
                status_code=result["status_code"],
                content=result["response"]
            )
        
        elif action == "index_markdown":
            result = index_markdown(task_info.get("params", {}))
            return JSONResponse(
                status_code=result["status_code"],
                content=result["response"]
            )
        elif action == "email_extraction":
            result = handle_email_extraction(task_info.get("params", {}))
            return JSONResponse(
                status_code=result["status_code"],
                content=result["response"]
            )
        elif action == "credit_card_number_extraction":
            result = handle_credit_card_number_extraction(task_info.get("params", {}))
            return JSONResponse(
                status_code=result["status_code"],
                content=result["response"]
            )
        
        elif action == "similar_comments":
            result = handle_similar_comments(task_info.get("params", {}))
            return JSONResponse(
                status_code=result["status_code"],
                content=result["response"]
            )
        elif action == "get_tickets_database_query":
            result = handle_database_query(task_info.get("params", {}))
            return JSONResponse(
                status_code=result["status_code"],
                content=result["response"]
            )
        
        elif action =="git_clone_repo":
            result=handle_clone_git_repo(task_info.get("params", {}))
            return JSONResponse(
                status_code=result["status_code"],
                content=result["response"]
            )
        elif action == "scrape_website":
            result = handle_scrape_website(task_info.get("params", {}))
            return JSONResponse(
        status_code=result["status_code"],
        content=result["response"])

        elif action == "compress_resize_image":
            result = handle_compress_resize_image(task_info.get("params", {}))
            return JSONResponse(
                status_code=result["status_code"],
                content=result["response"]
            )
        elif action =="transcribe_audio_file":
            result=handle_transcribe_audio(task_info.get("params", {}))
            return JSONResponse(
                status_code=result["status_code"],
                content=result["response"]
            )
        elif action == "convert_markdown_to_html":
            result = handle_convert_markdown_to_html(task_info.get("params", {}))
            return JSONResponse(
                status_code=result["status_code"],
                content=result["response"]
            )
        
        elif action =="markdown_to_html":
            result=handle_convert_markdown_to_html(task_info.get("params", {}))
            return JSONResponse(
                status_code=result["status_code"],
                content=result["response"]
            )
        elif action == "transcribe_audio_file":
            result=handle_transcribe_audio(task_info.get("params", {}))
            return JSONResponse(
                status_code=result["status_code"],
                content=result["response"]
            )
        elif action == "filter_csv_file":
            result = handle_filter_csv(task_info.get("params", {}))
            return JSONResponse(
                status_code=result["status_code"],
                content=result["response"]
            )
        elif action == "translate_text":
            result = handle_translate_text(task_info.get("params", {}))
            return JSONResponse(
                status_code=result["status_code"],
                content=result["response"]
            )
        elif action == "fetch_external_api":
            result = handle_external_api_request(task_info.get("params", {}))
            return JSONResponse(
                status_code=result["status_code"],
                content=result["response"]
            )
        
        else:
            return JSONResponse(
                status_code=400,
                content={
                    "success": False,
                    "message": f"Unknown action: {action}",
                    "error": "Invalid task type",
                    "status": 400
                }
            )
            
    except json.JSONDecodeError:
        return JSONResponse(
            status_code=400,
            content={
                "success": False,
                "message": "Invalid task format",
                "error": "Could not parse task JSON",
                "status": 400
            }
        )
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
                "success": False,
                "message": "Internal server error",
                "error": str(e),
                "status": 500
            }
        )

@app.get("/read",
    response_model=ReadFileResponse,
    responses={
        200: {
            "description": "File read successfully",
            "model": ReadFileResponse,
            "content": {
                "text/plain": {
                    "example": "example.txt"   
                    ""             
                }
            }
        },
        403: {
            "description": "Access forbidden",
            "model": BaseResponse,
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "message": "Access forbidden",
                        "error": "Path outside DATA_DIR",
                        "status": 403
                    }
                }
            }
        },
        404: {
            "description": "File not found",
            "model": BaseResponse,
            "content": {
                "application/json": {
                    "example": {
                        "success": False,
                        "message": "File not found",
                        "error": "File example.txt does not exist",
                        "status": 404
                    }
                }
            }
        }
    }
)
async def read_file(path: str = Query(..., description="File path to read")):
    """Read and return the contents of a file."""
    try:
        # Check if /data/ prefix is present and remove if it is
        if '/data/' in path:
            cleaned_path = path.replace('/data/', '').lstrip("/")
        elif 'data/' in path:
            cleaned_path = path.replace('data/', '').lstrip("/")
        else:
            cleaned_path = path.lstrip("/")
        
        full_path = os.path.join(DATA_DIR, cleaned_path)
        if not os.path.abspath(full_path).startswith(os.path.abspath(DATA_DIR)):
            return JSONResponse(
            status_code=403,
            content={
                "success": False,
                "message": "Access forbidden",
                "error": "Path outside DATA_DIR",
                "status": 403
            }
            )
            
        if not os.path.exists(full_path):
            return JSONResponse(
            status_code=404,
            content={
                "success": False,
                "message": "File not found", 
                "error": f"File {cleaned_path} does not exist",
                "status": 404
            }
            )
        
        with open(full_path, "r") as f:
            return JSONResponse(
            status_code=200,
            content=f.read().strip().strip('"').strip("'"),
            media_type="text/plain")
            
    except Exception as e:
        return JSONResponse(
            status_code=500,
            content={
            "success": False,
            "message": "Internal server error",
            "error": str(e),
            "status": 500
            }
        )


if __name__ == "__main__":
    uvicorn.run(app, host="127.0.0.1", port=8000)


