import tkinter as tk
from tkinter import ttk, scrolledtext, messagebox, filedialog
import customtkinter as ctk
import os
import json
import datetime
from pathlib import Path
import threading
from dotenv import load_dotenv

# Import our modules
from content_assistant import ContentAssistant
from filesystem_manager import FilesystemManager
from medium_publisher import MediumPublisher

# Load environment variables
load_dotenv()

class ContentCreationGUI:
    def __init__(self):
        # Set appearance mode and color theme
        ctk.set_appearance_mode("dark")
        ctk.set_default_color_theme("blue")
        
        # Initialize main window
        self.root = ctk.CTk()
        self.root.title("Content Creation Assistant")
        self.root.geometry("1200x800")
        
        # Initialize components
        self.filesystem_manager = FilesystemManager()
        self.content_assistant = ContentAssistant()
        self.medium_publisher = MediumPublisher()
        
        # Setup workspace
        self.setup_workspace()
        
        # Create GUI components
        self.create_gui()
        
        # Update file browser
        self.update_file_browser()
    
    def setup_workspace(self):
        """Create the required workspace directories"""
        workspace_dirs = [
            "content-workspace/ideas",
            "content-workspace/generated", 
            "content-workspace/published",
            "content-workspace/templates"
        ]
        
        for dir_path in workspace_dirs:
            Path(dir_path).mkdir(parents=True, exist_ok=True)
    
    def create_gui(self):
        """Create the main GUI layout"""
        # Create main frame
        main_frame = ctk.CTkFrame(self.root)
        main_frame.pack(fill="both", expand=True, padx=10, pady=10)
        
        # Create left panel (Chat Interface)
        self.create_chat_panel(main_frame)
        
        # Create right panel (File Browser and Editor)
        self.create_file_panel(main_frame)
    
    def create_chat_panel(self, parent):
        """Create the chat interface panel"""
        chat_frame = ctk.CTkFrame(parent, width=400)
        chat_frame.pack(side="left", fill="both", expand=True, padx=(0, 5))
        
        # Chat title
        chat_title = ctk.CTkLabel(chat_frame, text="Content Assistant Chat", font=("Arial", 16, "bold"))
        chat_title.pack(pady=10)
        
        # Chat display area
        self.chat_display = ctk.CTkTextbox(chat_frame, height=500)
        self.chat_display.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Input frame
        input_frame = ctk.CTkFrame(chat_frame)
        input_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        # Chat input
        self.chat_input = ctk.CTkEntry(input_frame, placeholder_text="Type your message here...")
        self.chat_input.pack(side="left", fill="x", expand=True, padx=(0, 5))
        self.chat_input.bind("<Return>", self.send_chat_message)
        
        # Send button
        send_button = ctk.CTkButton(input_frame, text="Send", command=self.send_chat_message)
        send_button.pack(side="right")
        
        # Quick action buttons
        self.create_quick_actions(chat_frame)
    
    def create_quick_actions(self, parent):
        """Create quick action buttons"""
        actions_frame = ctk.CTkFrame(parent)
        actions_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        # Quick action buttons
        ctk.CTkButton(actions_frame, text="Capture Idea", command=self.capture_idea).pack(fill="x", pady=2)
        ctk.CTkButton(actions_frame, text="Generate Article", command=self.generate_article).pack(fill="x", pady=2)
        ctk.CTkButton(actions_frame, text="Publish to Medium", command=self.publish_to_medium).pack(fill="x", pady=2)
    
    def create_file_panel(self, parent):
        """Create the file browser and editor panel"""
        file_frame = ctk.CTkFrame(parent, width=600)
        file_frame.pack(side="right", fill="both", expand=True, padx=(5, 0))
        
        # File browser title
        file_title = ctk.CTkLabel(file_frame, text="Content Workspace", font=("Arial", 16, "bold"))
        file_title.pack(pady=10)
        
        # File browser
        self.create_file_browser(file_frame)
        
        # Content editor
        self.create_content_editor(file_frame)
    
    def create_file_browser(self, parent):
        """Create the file browser"""
        browser_frame = ctk.CTkFrame(parent)
        browser_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        # Browser title
        browser_label = ctk.CTkLabel(browser_frame, text="Files:")
        browser_label.pack(anchor="w", padx=10, pady=(10, 5))
        
        # File tree
        self.file_tree = ttk.Treeview(browser_frame, height=8)
        self.file_tree.pack(fill="x", padx=10, pady=(0, 10))
        self.file_tree.bind("<Double-1>", self.on_file_select)
        
        # Refresh button
        refresh_button = ctk.CTkButton(browser_frame, text="Refresh", command=self.update_file_browser)
        refresh_button.pack(pady=(0, 10))
    
    def create_content_editor(self, parent):
        """Create the content editor"""
        editor_frame = ctk.CTkFrame(parent)
        editor_frame.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Editor title
        editor_label = ctk.CTkLabel(editor_frame, text="Content Editor:")
        editor_label.pack(anchor="w", padx=10, pady=(10, 5))
        
        # Content editor
        self.content_editor = ctk.CTkTextbox(editor_frame)
        self.content_editor.pack(fill="both", expand=True, padx=10, pady=(0, 10))
        
        # Editor buttons
        button_frame = ctk.CTkFrame(editor_frame)
        button_frame.pack(fill="x", padx=10, pady=(0, 10))
        
        ctk.CTkButton(button_frame, text="Save", command=self.save_content).pack(side="left", padx=(0, 5))
        ctk.CTkButton(button_frame, text="Move to Published", command=self.move_to_published).pack(side="left", padx=5)
    
    def send_chat_message(self, event=None):
        """Send a chat message and get AI response"""
        message = self.chat_input.get().strip()
        if not message:
            return
        
        # Display user message
        self.add_chat_message(f"You: {message}", "user")
        self.chat_input.delete(0, tk.END)
        
        # Process message in a separate thread
        threading.Thread(target=self.process_chat_message, args=(message,), daemon=True).start()
    
    def process_chat_message(self, message):
        """Process chat message and get AI response"""
        try:
            # Check for specific commands
            if "capture idea" in message.lower() or "write about" in message.lower():
                response = self.content_assistant.capture_idea(message)
            elif "generate article" in message.lower() or "generate from" in message.lower():
                response = self.content_assistant.generate_article(message)
            elif "publish" in message.lower():
                response = self.medium_publisher.publish_content(message)
            else:
                response = self.content_assistant.chat_response(message)
            
            # Display AI response
            self.root.after(0, lambda: self.add_chat_message(f"Assistant: {response}", "assistant"))
            
        except Exception as e:
            error_msg = f"Error: {str(e)}"
            self.root.after(0, lambda: self.add_chat_message(f"Assistant: {error_msg}", "error"))
    
    def add_chat_message(self, message, sender):
        """Add a message to the chat display"""
        self.chat_display.insert(tk.END, message + "\n\n")
        self.chat_display.see(tk.END)
    
    def update_file_browser(self):
        """Update the file browser with current workspace files"""
        # Clear existing items
        for item in self.file_tree.get_children():
            self.file_tree.delete(item)
        
        # Add workspace structure
        workspace_path = Path("content-workspace")
        if workspace_path.exists():
            self.add_directory_to_tree("", workspace_path)
    
    def add_directory_to_tree(self, parent, path):
        """Recursively add directory contents to tree"""
        try:
            for item in path.iterdir():
                if item.is_file():
                    self.file_tree.insert(parent, "end", text=item.name, values=(str(item),))
                elif item.is_dir():
                    folder = self.file_tree.insert(parent, "end", text=item.name, values=(str(item),))
                    self.add_directory_to_tree(folder, item)
        except PermissionError:
            pass
    
    def on_file_select(self, event):
        """Handle file selection in the browser"""
        selection = self.file_tree.selection()
        if selection:
            file_path = self.file_tree.item(selection[0])["values"][0]
            if os.path.isfile(file_path):
                self.load_file_content(file_path)
    
    def load_file_content(self, file_path):
        """Load file content into the editor"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            self.content_editor.delete("1.0", tk.END)
            self.content_editor.insert("1.0", content)
        except Exception as e:
            messagebox.showerror("Error", f"Could not load file: {str(e)}")
    
    def save_content(self):
        """Save content from editor to current file"""
        # This would need to track the current file being edited
        messagebox.showinfo("Info", "Save functionality to be implemented")
    
    def move_to_published(self):
        """Move current file from generated to published"""
        messagebox.showinfo("Info", "Move to published functionality to be implemented")
    
    def capture_idea(self):
        """Quick action to capture an idea"""
        self.chat_input.insert(0, "I want to capture an idea about ")
        self.chat_input.focus()
    
    def generate_article(self):
        """Quick action to generate an article"""
        self.chat_input.insert(0, "Generate article from ")
        self.chat_input.focus()
    
    def publish_to_medium(self):
        """Quick action to publish to Medium"""
        self.chat_input.insert(0, "Publish to Medium ")
        self.chat_input.focus()
    
    def run(self):
        """Start the GUI application"""
        self.root.mainloop()

if __name__ == "__main__":
    app = ContentCreationGUI()
    app.run() 