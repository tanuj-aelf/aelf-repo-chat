#!/usr/bin/env python3
import os
import json
import logging
import requests
import hashlib
import base64
import abc
import typing as t
import time
import threading
from flask import Flask, request, jsonify
from dotenv import load_dotenv
from Crypto.Cipher import AES

# Determine the project root directory (one level up from services/)
PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))

# Load environment variables
load_dotenv(os.path.join(PROJECT_ROOT, ".env"))

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
    handlers=[
        logging.FileHandler(os.path.join(PROJECT_ROOT, "logs", "lark_bot.log")),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger("lark_bot")

# Load Lark bot configuration
APP_ID = os.getenv("LARK_APP_ID")
APP_SECRET = os.getenv("LARK_APP_SECRET")
VERIFICATION_TOKEN = os.getenv("LARK_VERIFICATION_TOKEN")
ENCRYPT_KEY = os.getenv("LARK_ENCRYPT_KEY")
LARK_HOST = os.getenv("LARK_HOST", "https://open.feishu.cn")

# Constants for API endpoints
TENANT_ACCESS_TOKEN_URI = "/open-apis/auth/v3/tenant_access_token/internal"
MESSAGE_URI = "/open-apis/im/v1/messages"

# Setup simple file-based message deduplication
PROCESSED_MESSAGES_FILE = os.path.join(PROJECT_ROOT, "logs", "processed_messages.txt")
# Ensure the directory exists
os.makedirs(os.path.dirname(PROCESSED_MESSAGES_FILE), exist_ok=True)
# Ensure the file exists
if not os.path.exists(PROCESSED_MESSAGES_FILE):
    open(PROCESSED_MESSAGES_FILE, 'a').close()

# Simple message cache using a file with one ID per line
def is_message_processed(message_id):
    """Check if a message has already been processed by checking a flat file"""
    try:
        with open(PROCESSED_MESSAGES_FILE, 'r') as f:
            processed_ids = [line.strip() for line in f.readlines()]
            return message_id in processed_ids
    except Exception as e:
        logger.error(f"Error checking processed messages: {str(e)}")
        return False

def mark_message_processed(message_id):
    """Mark a message as processed by appending its ID to a flat file"""
    try:
        with open(PROCESSED_MESSAGES_FILE, 'a') as f:
            f.write(f"{message_id}\n")
        # Periodically clean up old messages to keep file size manageable
        cleanup_processed_messages()
    except Exception as e:
        logger.error(f"Error marking message as processed: {str(e)}")

def cleanup_processed_messages():
    """Keep only the most recent 1000 message IDs to manage file size"""
    try:
        # Only clean up if we reach a threshold (to avoid too many file operations)
        if os.path.getsize(PROCESSED_MESSAGES_FILE) > 100 * 1024:  # 100KB
            with open(PROCESSED_MESSAGES_FILE, 'r') as f:
                lines = f.readlines()
            
            # Keep only the most recent 1000 lines
            if len(lines) > 1000:
                with open(PROCESSED_MESSAGES_FILE, 'w') as f:
                    f.writelines(lines[-1000:])
                logger.info(f"Cleaned up processed messages file, kept {min(1000, len(lines))} entries")
    except Exception as e:
        logger.error(f"Error cleaning up processed messages: {str(e)}")

#-------------------------------------------------------
# Utils and Helper Classes (from the original project)
#-------------------------------------------------------

class Obj(dict):
    """Convert dict to object for easier attribute access"""
    def __init__(self, d):
        for a, b in d.items():
            if isinstance(b, (list, tuple)):
                setattr(self, a, [Obj(x) if isinstance(x, dict) else x for x in b])
            else:
                setattr(self, a, Obj(b) if isinstance(b, dict) else b)

def dict_2_obj(d: dict):
    """Convert dictionary to object"""
    return Obj(d)

class AESCipher(object):
    """AES encryption/decryption for Lark messages"""
    def __init__(self, key):
        self.bs = AES.block_size
        self.key = hashlib.sha256(AESCipher.str_to_bytes(key)).digest()

    @staticmethod
    def str_to_bytes(data):
        u_type = type(b"".decode("utf8"))
        if isinstance(data, u_type):
            return data.encode("utf8")
        return data

    @staticmethod
    def _unpad(s):
        return s[: -ord(s[len(s) - 1 :])]

    def decrypt(self, enc):
        iv = enc[: AES.block_size]
        cipher = AES.new(self.key, AES.MODE_CBC, iv)
        return self._unpad(cipher.decrypt(enc[AES.block_size :]))

    def decrypt_string(self, enc):
        enc = base64.b64decode(enc)
        return self.decrypt(enc).decode("utf8")

#-------------------------------------------------------
# Message API Client
#-------------------------------------------------------

class LarkException(Exception):
    """Exception for Lark API errors"""
    def __init__(self, code=0, msg=None):
        self.code = code
        self.msg = msg

    def __str__(self) -> str:
        return "{}:{}".format(self.code, self.msg)

    __repr__ = __str__

class MessageApiClient(object):
    """Client for sending messages to Lark"""
    def __init__(self, app_id, app_secret, lark_host):
        self._app_id = app_id
        self._app_secret = app_secret
        self._lark_host = lark_host
        self._tenant_access_token = ""

    @property
    def tenant_access_token(self):
        return self._tenant_access_token

    def send_text_with_open_id(self, open_id, content):
        """Send text message to user by open_id"""
        self.send("open_id", open_id, "text", content)

    def send(self, receive_id_type, receive_id, msg_type, content):
        """Send message to user"""
        self._authorize_tenant_access_token()
        url = "{}{}?receive_id_type={}".format(
            self._lark_host, MESSAGE_URI, receive_id_type
        )
        headers = {
            "Content-Type": "application/json",
            "Authorization": "Bearer " + self.tenant_access_token,
        }

        req_body = {
            "receive_id": receive_id,
            "content": content,
            "msg_type": msg_type,
        }
        resp = requests.post(url=url, headers=headers, json=req_body)
        MessageApiClient._check_error_response(resp)

    def _authorize_tenant_access_token(self):
        """Get tenant_access_token and set it for requests"""
        url = "{}{}".format(self._lark_host, TENANT_ACCESS_TOKEN_URI)
        req_body = {"app_id": self._app_id, "app_secret": self._app_secret}
        response = requests.post(url, req_body)
        MessageApiClient._check_error_response(response)
        self._tenant_access_token = response.json().get("tenant_access_token")

    @staticmethod
    def _check_error_response(resp):
        """Check if the response contains error information"""
        if resp.status_code != 200:
            resp.raise_for_status()
        response_dict = resp.json()
        code = response_dict.get("code", -1)
        if code != 0:
            logging.error(response_dict)
            raise LarkException(code=code, msg=response_dict.get("msg"))

#-------------------------------------------------------
# Event Handler Classes
#-------------------------------------------------------

class InvalidEventException(Exception):
    """Exception for invalid events"""
    def __init__(self, error_info):
        self.error_info = error_info

    def __str__(self) -> str:
        return "Invalid event: {}".format(self.error_info)

    __repr__ = __str__

class Event(object):
    """Base class for events"""
    callback_handler = None

    def __init__(self, dict_data, token, encrypt_key):
        # Event check and init
        header = dict_data.get("header")
        event = dict_data.get("event")
        if header is None or event is None:
            raise InvalidEventException("request is not callback event(v2)")
        self.header = dict_2_obj(header)
        self.event = dict_2_obj(event)
        self._validate(token, encrypt_key)

    def _validate(self, token, encrypt_key):
        """Validate event signature"""
        if self.header.token != token:
            raise InvalidEventException("invalid token")
        timestamp = request.headers.get("X-Lark-Request-Timestamp")
        nonce = request.headers.get("X-Lark-Request-Nonce")
        signature = request.headers.get("X-Lark-Signature")
        body = request.data
        bytes_b1 = (timestamp + nonce + encrypt_key).encode("utf-8")
        bytes_b = bytes_b1 + body
        h = hashlib.sha256(bytes_b)
        if signature != h.hexdigest():
            raise InvalidEventException("invalid signature in event")

    @abc.abstractmethod
    def event_type(self):
        return self.header.event_type

class MessageReceiveEvent(Event):
    """Event for received messages"""
    @staticmethod
    def event_type():
        return "im.message.receive_v1"

class UrlVerificationEvent(Event):
    """Event for URL verification"""
    def __init__(self, dict_data):
        self.event = dict_2_obj(dict_data)

    @staticmethod
    def event_type():
        return "url_verification"

class EventManager(object):
    """Manager for handling events"""
    event_callback_map = dict()
    event_type_map = dict()
    _event_list = [MessageReceiveEvent, UrlVerificationEvent]

    def __init__(self):
        for event in EventManager._event_list:
            EventManager.event_type_map[event.event_type()] = event

    def register(self, event_type: str) -> t.Callable:
        """Register a handler for an event type"""
        def decorator(f: t.Callable) -> t.Callable:
            self.register_handler_with_event_type(event_type=event_type, handler=f)
            return f
        return decorator

    @staticmethod
    def register_handler_with_event_type(event_type, handler):
        EventManager.event_callback_map[event_type] = handler

    @staticmethod
    def get_handler_with_event(token, encrypt_key):
        """Get the handler and event object for a request"""
        dict_data = json.loads(request.data)
        dict_data = EventManager._decrypt_data(encrypt_key, dict_data)
        callback_type = dict_data.get("type")
        
        # Only verification data has callback_type, else is event
        if callback_type == "url_verification":
            event = UrlVerificationEvent(dict_data)
            return EventManager.event_callback_map.get(event.event_type()), event

        # Only handle event v2
        schema = dict_data.get("schema")
        if schema is None:
            raise InvalidEventException("request is not callback event(v2)")

        # Get event_type
        event_type = dict_data.get("header").get("event_type")
        # Build event
        event = EventManager.event_type_map.get(event_type)(dict_data, token, encrypt_key)
        # Get handler
        return EventManager.event_callback_map.get(event_type), event

    @staticmethod
    def _decrypt_data(encrypt_key, data):
        """Decrypt data if needed"""
        encrypt_data = data.get("encrypt")
        if encrypt_key == "" and encrypt_data is None:
            # Data haven't been encrypted
            return data
        if encrypt_key == "":
            raise Exception("ENCRYPT_KEY is necessary")
        cipher = AESCipher(encrypt_key)
        return json.loads(cipher.decrypt_string(encrypt_data))

# Initialize API client for Lark bot
message_api_client = MessageApiClient(APP_ID, APP_SECRET, LARK_HOST)
event_manager = EventManager()

#-------------------------------------------------------
# Lark Bot Initialization
#-------------------------------------------------------

def init_lark_bot(app):
    """
    Initialize the Lark bot handler
    
    Args:
        app: The Flask app instance from main.py
    """
    # Register URL verification handler
    @event_manager.register("url_verification")
    def request_url_verify_handler(req_data: UrlVerificationEvent):
        # url verification, just need return challenge
        if req_data.event.token != VERIFICATION_TOKEN:
            raise InvalidEventException("VERIFICATION_TOKEN is invalid")
        return jsonify({"challenge": req_data.event.challenge})

    # Register message receive handler
    @event_manager.register("im.message.receive_v1")
    def message_receive_event_handler(req_data: MessageReceiveEvent):
        sender_id = req_data.event.sender.sender_id
        message = req_data.event.message
        if message.message_type != "text":
            logging.warn("Other types of messages have not been processed yet")
            return jsonify()
            
        # Get open_id and text_content
        open_id = sender_id.open_id
        message_id = message.message_id  # Get the message ID for deduplication
        text_content = json.loads(message.content).get("text", "")
        logger.info(f"Received message with ID {message_id} from {open_id}: {text_content}")
        
        # Check if message has already been processed - SIMPLE APPROACH
        if is_message_processed(message_id):
            logger.info(f"Duplicate message detected with ID {message_id}, ignoring")
            return jsonify({"status": "already_processed"})
        
        # Immediately mark the message as processed to prevent duplicate processing
        mark_message_processed(message_id)
        
        # Set a timeout for the entire handling process
        # Message response timeout
        MAX_PROCESSING_TIME = 60  # 60 seconds max processing time
        processing_complete = False
        processing_failed = False
        
        def process_message():
            nonlocal processing_complete
            nonlocal processing_failed
            
            try:
                # Process the query using the aelf-repo-chat agent
                # We'll make an internal request to the chat API
                headers = {"Content-Type": "application/json"}
                payload = {"query": text_content}
                
                logger.info("Sending request to chat API endpoint")
                # Make request to the chat API endpoint with timeout
                host = os.getenv("HOST", "0.0.0.0")
                port = os.getenv("PORT", 5000)
                response = requests.post(
                    f"http://{host}:{port}/api/chat", 
                    headers=headers, 
                    json=payload,
                    timeout=50  # 50 second timeout (less than the overall process timeout)
                )
                
                logger.info(f"Received response from chat API: status={response.status_code}")
                
                if response.status_code == 200:
                    # Extract the answer from the response
                    response_data = response.json()
                    answer = response_data.get("answer", "Sorry, I couldn't generate a response.")
                    
                    # Format sources for display
                    sources = response_data.get("source_documents", [])
                    if sources:
                        source_text = "\n\n**Sources:**\n"
                        
                        # GitHub URLs and repository information
                        github_base_url = "https://github.com"
                        default_owner = os.getenv("GITHUB_DEFAULT_OWNER", "AElfProject")
                        default_branch = os.getenv("GITHUB_DEFAULT_BRANCH", "main")
                        
                        for i, source in enumerate(sources[:3], 1):  # Limit to top 3 sources
                            file_path = source.get('source', 'unknown')
                            collection = source.get('collection', 'unknown')
                            relevance = source.get('relevance_score', 0)
                            
                            # Extract repository name from collection name
                            # Collection name format is typically "{repo-name}_collection"
                            repo_name = collection.replace("_collection", "") if collection.endswith("_collection") else collection
                            
                            # Read repository configuration to get correct branch
                            branch = default_branch
                            owner = default_owner
                            
                            try:
                                # Try to get repository configuration from config.json
                                config_path = os.path.join(PROJECT_ROOT, "config.json")
                                if os.path.exists(config_path):
                                    with open(config_path, "r") as f:
                                        config = json.load(f)
                                    
                                    # Find the repository in config
                                    repositories = config.get("repositories", [])
                                    for repo in repositories:
                                        if repo.get("name", "") == repo_name:
                                            # Use specified branch if available
                                            branch = repo.get("branch", default_branch)
                                            # Check if owner is specified in the URL
                                            repo_url = repo.get("url", "")
                                            if repo_url.startswith("https://github.com/"):
                                                url_parts = repo_url.replace("https://github.com/", "").split("/")
                                                if len(url_parts) >= 1:
                                                    owner = url_parts[0]
                            except Exception as e:
                                logger.warning(f"Error reading repository configuration: {str(e)}")
                            
                            # Handle repo names that might include owner (e.g., "owner-repo")
                            if "-" in repo_name and owner == default_owner:  # Only try to extract if we're using the default owner
                                parts = repo_name.split("-", 1)
                                # Only consider it an owner-repo format if it's likely
                                if len(parts) == 2 and len(parts[0]) > 2:
                                    # This might be in the format "owner-repo"
                                    # For example "aevatar-gagents" where "aevatar" could be the owner
                                    potential_owner = parts[0]
                                    potential_repo = parts[1]
                                    
                                    # If it's a known organization name, use it as owner
                                    known_orgs = ["aelf", "aevatar", "AElfProject"]
                                    if potential_owner.lower() in [org.lower() for org in known_orgs]:
                                        owner = potential_owner
                                        repo_name = potential_repo
                            
                            # Construct GitHub URL
                            github_url = f"{github_base_url}/{owner}/{repo_name}/blob/{branch}/{file_path}"
                            
                            # Format as markdown link for Lark
                            # Use formatted relevance score with 2 decimal places
                            formatted_relevance = f"{relevance:.2f}" if isinstance(relevance, float) else str(relevance)
                            source_text += f"{i}. [{file_path}]({github_url}) (relevance: {formatted_relevance})\n"
                        
                        answer += source_text
                    
                    # Send the response back to the user via Lark
                    logger.info("Sending response back to Lark user")
                    message_api_client.send_text_with_open_id(open_id, json.dumps({"text": answer}))
                    logger.info("Response sent successfully to Lark user")
                else:
                    error_msg = f"Error processing request: {response.status_code} - {response.text}"
                    logger.error(error_msg)
                    message_api_client.send_text_with_open_id(open_id, json.dumps({"text": "Sorry, I encountered an error processing your request."}))
                
                processing_complete = True
                    
            except requests.Timeout:
                logger.error("Request to chat API timed out")
                message_api_client.send_text_with_open_id(open_id, json.dumps({
                    "text": "Sorry, it's taking longer than expected to process your request. Please try again or simplify your query."
                }))
                processing_failed = True
                
            except Exception as e:
                logger.error(f"Error processing message: {str(e)}")
                # Send error message back to the user
                try:
                    message_api_client.send_text_with_open_id(open_id, json.dumps({
                        "text": "Sorry, I encountered an error processing your request."
                    }))
                except Exception as send_error:
                    logger.error(f"Failed to send error message to user: {str(send_error)}")
                
                processing_failed = True
        
        # Start message processing in a separate thread
        process_thread = threading.Thread(target=process_message)
        process_thread.start()
        
        # Wait for processing to complete, but with a timeout
        start_time = time.time()
        while not (processing_complete or processing_failed) and (time.time() - start_time < MAX_PROCESSING_TIME):
            time.sleep(0.5)
        
        # If processing is taking too long, send a timeout message
        if not (processing_complete or processing_failed):
            logger.error("Message processing timeout reached")
            try:
                message_api_client.send_text_with_open_id(open_id, json.dumps({
                    "text": "I'm sorry, but it's taking too long to process your request. Our systems might be busy. Please try again in a moment."
                }))
            except Exception as e:
                logger.error(f"Failed to send timeout message: {str(e)}")
        
        # Return a response to Lark regardless of the processing status
        # This ensures the Lark platform knows we received the message
        return jsonify({"status": "received"})

    # Register error handler
    @app.errorhandler(Exception)
    def lark_error_handler(ex):
        logger.error(f"Lark bot error: {str(ex)}")
        response = jsonify(message=str(ex))
        response.status_code = (
            ex.response.status_code if isinstance(ex, requests.HTTPError) else 500
        )
        return response

    # Register event callback endpoint
    # The endpoint function that handles Lark bot events
    def handle_lark_event():
        try:
            # Initialize callback instance and handle
            event_handler, event = event_manager.get_handler_with_event(VERIFICATION_TOKEN, ENCRYPT_KEY)
            
            if event_handler is None:
                logger.error(f"No handler for event type: {event.header.event_type if hasattr(event, 'header') else 'unknown'}")
                return jsonify({"error": "Event type not supported"}), 400
                
            return event_handler(event)
        except Exception as e:
            logger.error(f"Error handling Lark event: {str(e)}")
            return jsonify({"error": str(e)}), 500
    
    # Register the handler for both routes
    app.route("/", methods=["POST"])(handle_lark_event)
    app.route("/lark/event", methods=["POST"])(handle_lark_event)

