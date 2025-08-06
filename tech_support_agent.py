# Technical Support Chat Agent with Groq and Hybrid Retrieval
# Enhanced with BM25 + Dense Embeddings using RRF

import os
import json
import re
import pickle
import numpy as np
from typing import Dict, List, Optional, Tuple, Any
from datetime import datetime
from enum import Enum
from dataclasses import dataclass
import logging

# For Groq API
try:
    from groq import Groq
except ImportError:
    print("Groq not installed. Install with: pip install groq")

# For embeddings
try:
    from sentence_transformers import SentenceTransformer
except ImportError:
    print("Sentence transformers not installed. Install with: pip install sentence-transformers")

# For BM25
try:
    from rank_bm25 import BM25Okapi
except ImportError:
    print("rank-bm25 not installed. Install with: pip install rank-bm25")

# For vector operations
try:
    import faiss
except ImportError:
    print("Faiss not installed. Install with: pip install faiss-cpu")

# Additional imports
import requests
from pathlib import Path
import pandas as pd
from collections import defaultdict

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class IssueCategory(Enum):
    """Categories for technical issues"""
    SYSTEM = "System"
    NETWORK = "Network"
    SETTINGS = "Settings"
    HARDWARE = "Hardware"
    SOFTWARE = "Software"
    SECURITY = "Security"
    GENERAL = "General"
    UNKNOWN = "Unknown"

@dataclass
class Document:
    """Document structure for knowledge base"""
    id: str
    title: str
    content: str
    category: str
    keywords: List[str]
    solutions: List[str]
    metadata: Dict[str, Any]

class KnowledgeBaseBuilder:
    """Build and manage knowledge base from various sources"""
    
    @staticmethod
    def fetch_from_microsoft_docs():
        """
        Fetch real documentation from Microsoft Learn
        Uses the microsoft_learn_fetcher module for actual API calls
        """
        try:
            # Try to import the enhanced fetcher
            from microsoft_learn_fetcher import EnhancedKnowledgeBaseBuilder
            logger.info("Using Microsoft Learn API fetcher...")
            return EnhancedKnowledgeBaseBuilder.fetch_from_microsoft_docs()
        except ImportError:
            logger.warning("Microsoft Learn fetcher not available, using fallback data...")
    @staticmethod
    def _get_fallback_microsoft_docs():
        """
        Fetch technical documentation from Microsoft Learn
        Note: In production, use official APIs or scraping with permission
        """
        kb_data = {
            "network_troubleshooting": {
                "title": "Network Connectivity Troubleshooting",
                "content": "Complete guide for diagnosing and fixing network issues including Wi-Fi problems, ethernet connections, and internet connectivity.",
                "category": "Network",
                "keywords": ["wifi", "wi-fi", "internet", "connection", "network", "ethernet", "lan", "wan", "connectivity", "wireless"],
                "solutions": [
                    "Check physical connections and cables",
                    "Verify Wi-Fi is enabled and airplane mode is off",
                    "Run Windows Network Troubleshooter: Settings > Network & Internet > Status > Network troubleshooter",
                    "Reset network adapter: netsh winsock reset && netsh int ip reset",
                    "Update network drivers from Device Manager",
                    "Flush DNS cache: ipconfig /flushdns",
                    "Check router status and restart if needed",
                    "Verify DHCP is enabled or static IP is correct",
                    "Disable VPN or proxy if connected",
                    "Check Windows Firewall and antivirus settings"
                ]
            },
            "system_performance": {
                "title": "System Performance Optimization",
                "content": "Methods to improve system performance, fix slow computers, and optimize Windows settings.",
                "category": "System",
                "keywords": ["slow", "performance", "speed", "lag", "freeze", "hang", "crash", "memory", "cpu", "disk"],
                "solutions": [
                    "Check Task Manager for high CPU/Memory usage",
                    "Disable startup programs: Task Manager > Startup tab",
                    "Run Disk Cleanup: cleanmgr.exe",
                    "Check for malware with Windows Defender",
                    "Update Windows and drivers",
                    "Adjust visual effects: System Properties > Advanced > Performance Settings",
                    "Increase virtual memory if needed",
                    "Run SFC scan: sfc /scannow",
                    "Check hard drive health: chkdsk /f /r",
                    "Consider adding more RAM or upgrading to SSD"
                ]
            },
            "windows_update_issues": {
                "title": "Windows Update Troubleshooting",
                "content": "Resolve Windows Update errors, stuck updates, and installation failures.",
                "category": "System",
                "keywords": ["update", "windows update", "patch", "installation", "download", "error", "stuck", "failed"],
                "solutions": [
                    "Run Windows Update Troubleshooter",
                    "Clear Windows Update cache: Stop wuauserv, delete SoftwareDistribution folder",
                    "Reset Windows Update components using batch script",
                    "Check disk space (need at least 20GB free)",
                    "Temporarily disable antivirus during update",
                    "Use Windows Update Assistant tool",
                    "Check for corrupted system files: DISM /Online /Cleanup-Image /RestoreHealth",
                    "Download updates manually from Microsoft Update Catalog",
                    "Perform clean boot and try updating",
                    "Check Event Viewer for specific error codes"
                ]
            },
            "printer_issues": {
                "title": "Printer Setup and Troubleshooting",
                "content": "Fix common printer problems including connection issues, print queue problems, and driver issues.",
                "category": "Hardware",
                "keywords": ["printer", "print", "printing", "scanner", "queue", "spooler", "driver", "paper", "ink"],
                "solutions": [
                    "Check printer power and connections",
                    "Clear print queue: Services > Print Spooler > Stop/Start",
                    "Run Printer Troubleshooter from Settings",
                    "Update or reinstall printer drivers",
                    "Set correct default printer",
                    "Check printer status and error messages",
                    "Verify printer is on same network (for network printers)",
                    "Remove and re-add printer",
                    "Check for paper jams and ink levels",
                    "Reset printer to factory settings if needed"
                ]
            },
            "audio_problems": {
                "title": "Audio and Sound Troubleshooting",
                "content": "Resolve no sound, audio quality issues, and microphone problems.",
                "category": "Hardware",
                "keywords": ["sound", "audio", "speaker", "microphone", "headphone", "volume", "mute", "noise", "echo"],
                "solutions": [
                    "Check volume levels and mute status",
                    "Verify correct playback device is selected",
                    "Run Audio Troubleshooter from Settings",
                    "Update audio drivers from Device Manager",
                    "Check audio service status: Windows Audio",
                    "Disable audio enhancements",
                    "Check physical connections and cables",
                    "Test with different applications",
                    "Adjust sample rate and bit depth in Sound settings",
                    "Reinstall audio drivers if corrupted"
                ]
            },
            "display_issues": {
                "title": "Display and Graphics Problems",
                "content": "Fix screen resolution, multiple monitor setup, and graphics driver issues.",
                "category": "Hardware",
                "keywords": ["display", "monitor", "screen", "resolution", "graphics", "video", "hdmi", "dual", "blank"],
                "solutions": [
                    "Check cable connections (HDMI, DisplayPort, VGA)",
                    "Adjust display resolution: Right-click desktop > Display settings",
                    "Update graphics drivers from manufacturer website",
                    "Detect displays: Display settings > Detect",
                    "Change refresh rate if screen flickering",
                    "Disable/enable graphics adapter in Device Manager",
                    "Boot in Safe Mode to troubleshoot driver issues",
                    "Check monitor power and input source",
                    "Reset display settings to default",
                    "Run Display Quality Troubleshooter"
                ]
            },
            "browser_issues": {
                "title": "Web Browser Troubleshooting",
                "content": "Fix browser crashes, slow performance, and website loading issues.",
                "category": "Software",
                "keywords": ["browser", "chrome", "firefox", "edge", "internet explorer", "website", "cache", "cookies", "extension"],
                "solutions": [
                    "Clear browser cache and cookies",
                    "Disable browser extensions one by one",
                    "Reset browser to default settings",
                    "Update browser to latest version",
                    "Check for conflicting software",
                    "Disable hardware acceleration",
                    "Create new browser profile",
                    "Check proxy and firewall settings",
                    "Scan for malware and adware",
                    "Try different browser to isolate issue"
                ]
            },
            "email_configuration": {
                "title": "Email Setup and Issues",
                "content": "Configure email clients, fix sending/receiving issues, and resolve Outlook problems.",
                "category": "Software",
                "keywords": ["email", "outlook", "mail", "smtp", "imap", "pop3", "exchange", "calendar", "contacts"],
                "solutions": [
                    "Verify email server settings (SMTP, IMAP/POP)",
                    "Check username and password",
                    "Configure correct ports and encryption",
                    "Clear Outlook cache and offline files",
                    "Repair Outlook data file (PST/OST)",
                    "Create new Outlook profile",
                    "Check mailbox quota and size",
                    "Disable antivirus email scanning temporarily",
                    "Update email client to latest version",
                    "Check firewall rules for email ports"
                ]
            },
            "security_basics": {
                "title": "Basic Security and Privacy",
                "content": "Essential security practices, antivirus management, and privacy settings.",
                "category": "Security",
                "keywords": ["security", "virus", "malware", "antivirus", "firewall", "privacy", "password", "backup", "ransomware"],
                "solutions": [
                    "Enable Windows Defender real-time protection",
                    "Keep Windows and software updated",
                    "Use strong, unique passwords",
                    "Enable two-factor authentication where possible",
                    "Regular backups using File History or backup software",
                    "Configure Windows Firewall properly",
                    "Review privacy settings in Windows Settings",
                    "Be cautious with email attachments and links",
                    "Use standard user account, not administrator",
                    "Enable BitLocker for disk encryption"
                ]
            },
            "file_recovery": {
                "title": "File Recovery and Backup",
                "content": "Recover deleted files, restore from backups, and prevent data loss.",
                "category": "System",
                "keywords": ["recovery", "backup", "restore", "deleted", "lost", "file", "data", "recycle", "history"],
                "solutions": [
                    "Check Recycle Bin first",
                    "Use File History to restore previous versions",
                    "Try Windows File Recovery tool",
                    "Check for shadow copies",
                    "Use third-party recovery software if needed",
                    "Enable and configure File History",
                    "Set up regular automated backups",
                    "Use OneDrive or cloud backup services",
                    "Create system restore points regularly",
                    "Keep important files in multiple locations"
                ]
            }
        }
        
        return kb_data
    
    @staticmethod
    def fetch_from_stackoverflow():
        """
        Fetch common IT issues from Stack Overflow
        Note: Use Stack Exchange API in production
        """
        additional_kb = {
            "blue_screen": {
                "title": "Blue Screen of Death (BSOD) Resolution",
                "content": "Diagnose and fix Windows blue screen errors and system crashes.",
                "category": "System",
                "keywords": ["bsod", "blue screen", "crash", "error", "stop code", "memory dump", "system failure"],
                "solutions": [
                    "Note the stop code and error message",
                    "Boot into Safe Mode if possible",
                    "Check for recent hardware or software changes",
                    "Run memory diagnostic: mdsched.exe",
                    "Update or roll back drivers",
                    "Check system files: sfc /scannow",
                    "Review dump files with BlueScreenView",
                    "Test RAM with MemTest86",
                    "Check hard drive health",
                    "Perform system restore or reset if needed"
                ]
            },
            "remote_desktop": {
                "title": "Remote Desktop Connection Issues",
                "content": "Setup and troubleshoot Remote Desktop Protocol (RDP) connections.",
                "category": "Network",
                "keywords": ["remote", "rdp", "remote desktop", "terminal", "vnc", "teamviewer", "anydesk"],
                "solutions": [
                    "Enable Remote Desktop in System Properties",
                    "Check Windows Firewall rules for RDP (port 3389)",
                    "Verify user has remote access permissions",
                    "Check network connectivity between computers",
                    "Ensure Remote Desktop Services is running",
                    "Update Remote Desktop client",
                    "Check group policy settings",
                    "Verify RDP port if changed from default",
                    "Try IP address instead of computer name",
                    "Check for VPN requirements"
                ]
            }
        }
        return additional_kb
    
    @staticmethod
    def build_complete_kb():
        """Combine all knowledge sources"""
        kb = {}
        kb.update(KnowledgeBaseBuilder.fetch_from_microsoft_docs())
        kb.update(KnowledgeBaseBuilder.fetch_from_stackoverflow())
        
        # Convert to Document objects
        documents = []
        for doc_id, data in kb.items():
            doc = Document(
                id=doc_id,
                title=data["title"],
                content=data["content"],
                category=data["category"],
                keywords=data["keywords"],
                solutions=data["solutions"],
                metadata=data.get("metadata", {})
            )
            documents.append(doc)
        
        return documents

class HybridRetriever:
    """Hybrid retrieval using BM25 and dense embeddings with RRF"""
    
    def __init__(self, documents: List[Document], embedding_model: str = "all-MiniLM-L6-v2"):
        self.documents = documents
        self.embedding_model = SentenceTransformer(embedding_model)
        
        # Prepare corpus for BM25
        self.corpus = []
        self.doc_map = {}
        
        for i, doc in enumerate(documents):
            # Combine all text fields for retrieval
            text = f"{doc.title} {doc.content} {' '.join(doc.keywords)} {' '.join(doc.solutions)}"
            self.corpus.append(text.lower().split())
            self.doc_map[i] = doc
        
        # Initialize BM25
        self.bm25 = BM25Okapi(self.corpus)
        
        # Create dense embeddings
        self._create_dense_index()
    
    def _create_dense_index(self):
        """Create FAISS index for dense retrieval"""
        logger.info("Creating dense embeddings for knowledge base...")
        
        # Generate embeddings for all documents
        texts = [f"{doc.title} {doc.content}" for doc in self.documents]
        self.embeddings = self.embedding_model.encode(texts, show_progress_bar=True)
        
        # Create FAISS index
        dimension = self.embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)  # Inner product for cosine similarity
        
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(self.embeddings)
        self.index.add(self.embeddings)
        
        logger.info(f"Created FAISS index with {len(self.documents)} documents")
    
    def bm25_search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """BM25 search"""
        tokenized_query = query.lower().split()
        scores = self.bm25.get_scores(tokenized_query)
        
        # Get top k documents
        top_indices = np.argsort(scores)[::-1][:k]
        results = [(idx, scores[idx]) for idx in top_indices]
        
        return results
    
    def dense_search(self, query: str, k: int = 10) -> List[Tuple[int, float]]:
        """Dense embedding search using FAISS"""
        # Encode query
        query_embedding = self.embedding_model.encode([query])
        faiss.normalize_L2(query_embedding)
        
        # Search
        distances, indices = self.index.search(query_embedding, k)
        
        results = [(int(idx), float(score)) for idx, score in zip(indices[0], distances[0])]
        return results
    
    def hybrid_search(self, query: str, k: int = 5, alpha: float = 0.5) -> List[Document]:
        """
        Hybrid search using Reciprocal Rank Fusion (RRF)
        
        Args:
            query: Search query
            k: Number of results to return
            alpha: Weight for RRF (higher = more weight on early ranks)
        """
        # Get results from both methods
        bm25_results = self.bm25_search(query, k=k*2)
        dense_results = self.dense_search(query, k=k*2)
        
        # RRF scoring
        rrf_scores = defaultdict(float)
        
        # Add BM25 scores
        for rank, (idx, _) in enumerate(bm25_results):
            rrf_scores[idx] += 1.0 / (alpha + rank + 1)
        
        # Add dense scores
        for rank, (idx, _) in enumerate(dense_results):
            rrf_scores[idx] += 1.0 / (alpha + rank + 1)
        
        # Sort by RRF score
        sorted_docs = sorted(rrf_scores.items(), key=lambda x: x[1], reverse=True)
        
        # Return top k documents
        results = []
        for idx, score in sorted_docs[:k]:
            if idx in self.doc_map:
                results.append(self.doc_map[idx])
        
        return results
    
    def save_index(self, path: str):
        """Save the retriever state"""
        save_data = {
            'documents': self.documents,
            'corpus': self.corpus,
            'embeddings': self.embeddings
        }
        
        # Save FAISS index
        faiss.write_index(self.index, f"{path}_faiss.index")
        
        # Save other data
        with open(f"{path}_data.pkl", 'wb') as f:
            pickle.dump(save_data, f)
        
        logger.info(f"Saved retriever to {path}")
    
    @classmethod
    def load_index(cls, path: str, embedding_model: str = "all-MiniLM-L6-v2"):
        """Load saved retriever"""
        # Load data
        with open(f"{path}_data.pkl", 'rb') as f:
            save_data = pickle.load(f)
        
        # Create retriever
        retriever = cls(save_data['documents'], embedding_model)
        
        # Load FAISS index
        retriever.index = faiss.read_index(f"{path}_faiss.index")
        retriever.embeddings = save_data['embeddings']
        
        logger.info(f"Loaded retriever from {path}")
        return retriever

class TechSupportAgent:
    """Enhanced technical support agent with Groq and hybrid retrieval"""
    
    def __init__(self, groq_api_key: str, model_name: str = "mixtral-8x7b-32768"):
        """
        Initialize the tech support agent with Groq
        
        Args:
            groq_api_key: Groq API key
            model_name: Groq model to use (mixtral-8x7b-32768, llama2-70b-4096, etc.)
        """
        self.client = Groq(api_key=groq_api_key)
        self.model_name = model_name
        self.conversation_history = []
        
        # Build knowledge base
        logger.info("Building knowledge base...")
        documents = KnowledgeBaseBuilder.build_complete_kb()
        
        # Initialize hybrid retriever
        self.retriever = HybridRetriever(documents)
        
        # System prompt
        self.system_prompt = self._create_system_prompt()
    
    def _create_system_prompt(self) -> str:
        """Create the system prompt for the agent"""
        return """You are an expert IT support specialist with deep knowledge of Windows, macOS, and Linux systems. Your role is to provide clear, accurate, and helpful technical support.

Key responsibilities:
1. Diagnose technical issues accurately
2. Provide step-by-step solutions that are easy to follow
3. Explain technical concepts in simple terms
4. Ask clarifying questions when needed
5. Suggest preventive measures
6. Escalate to human support when appropriate

Guidelines:
- Be patient and understanding
- Use numbered steps for instructions
- Provide multiple solutions when available
- Include keyboard shortcuts and command-line options
- Warn about potential risks
- Suggest backups before major changes
- Stay professional and encouraging

You have access to a comprehensive knowledge base. Use it to provide accurate, specific solutions."""
    
    def classify_issue(self, user_input: str) -> IssueCategory:
        """Classify the user's issue into a category"""
        input_lower = user_input.lower()
        
        # Keyword-based classification
        network_keywords = ["wifi", "internet", "network", "connection", "ethernet", "ip", "dns", "router"]
        system_keywords = ["restart", "boot", "crash", "slow", "performance", "update", "bsod"]
        hardware_keywords = ["printer", "mouse", "keyboard", "monitor", "screen", "audio", "sound", "usb"]
        software_keywords = ["browser", "email", "outlook", "application", "program", "install"]
        security_keywords = ["virus", "malware", "security", "firewall", "password", "hack"]
        settings_keywords = ["setting", "configure", "time", "date", "language", "display"]
        
        if any(kw in input_lower for kw in network_keywords):
            return IssueCategory.NETWORK
        elif any(kw in input_lower for kw in system_keywords):
            return IssueCategory.SYSTEM
        elif any(kw in input_lower for kw in hardware_keywords):
            return IssueCategory.HARDWARE
        elif any(kw in input_lower for kw in software_keywords):
            return IssueCategory.SOFTWARE
        elif any(kw in input_lower for kw in security_keywords):
            return IssueCategory.SECURITY
        elif any(kw in input_lower for kw in settings_keywords):
            return IssueCategory.SETTINGS
        else:
            return IssueCategory.UNKNOWN
    
    def retrieve_context(self, query: str, k: int = 3) -> str:
        """Retrieve relevant context using hybrid search"""
        # Perform hybrid search
        relevant_docs = self.retriever.hybrid_search(query, k=k)
        
        if not relevant_docs:
            return ""
        
        # Format context
        context_parts = []
        for i, doc in enumerate(relevant_docs, 1):
            context_parts.append(f"### Resource {i}: {doc.title}")
            context_parts.append(f"Category: {doc.category}")
            context_parts.append(f"Description: {doc.content}")
            context_parts.append("Solutions:")
            for j, solution in enumerate(doc.solutions, 1):
                context_parts.append(f"  {j}. {solution}")
            context_parts.append("")
        
        return "\n".join(context_parts)
    
    def generate_response(self, user_input: str) -> Dict:
        """Generate response using Groq with RAG"""
        # Classify issue
        category = self.classify_issue(user_input)
        
        # Retrieve relevant context
        context = self.retrieve_context(user_input)
        
        # Prepare messages
        messages = [
            {"role": "system", "content": self.system_prompt}
        ]
        
        # Add conversation history (keep last 4 exchanges)
        for msg in self.conversation_history[-8:]:
            messages.append(msg)
        
        # Add current query with context
        user_message = user_input
        if context:
            user_message = f"""Based on the following technical resources, please help with the user's issue.

Technical Resources:
{context}

User Issue: {user_input}

Please provide a clear, step-by-step solution based on the resources above, adapting them to the user's specific situation."""
        
        messages.append({"role": "user", "content": user_message})
        
        try:
            # Call Groq API
            completion = self.client.chat.completions.create(
                model=self.model_name,
                messages=messages,
                temperature=0.3,  # Lower temperature for more consistent technical advice
                max_tokens=1000,
                top_p=0.9,
                stream=False
            )
            
            response = completion.choices[0].message.content
            
            # Update conversation history
            self.conversation_history.append({"role": "user", "content": user_input})
            self.conversation_history.append({"role": "assistant", "content": response})
            
            return {
                "response": response,
                "category": category.value,
                "has_context": bool(context),
                "model": self.model_name,
                "timestamp": datetime.now().isoformat()
            }
            
        except Exception as e:
            logger.error(f"Error generating response: {str(e)}")
            return {
                "response": f"I apologize, but I encountered an error: {str(e)}. Please try again or contact support.",
                "category": category.value,
                "has_context": bool(context),
                "error": str(e),
                "timestamp": datetime.now().isoformat()
            }
    
    def reset_conversation(self):
        """Reset conversation history"""
        self.conversation_history = []
        logger.info("Conversation history reset")

# CLI Interface
class CLIInterface:
    """Command-line interface for the chat agent"""
    
    def __init__(self, agent: TechSupportAgent):
        self.agent = agent
        self.running = True
    
    def display_welcome(self):
        """Display welcome message"""
        print("\n" + "="*70)
        print("🤖 Advanced Technical Support Assistant (Powered by Groq)")
        print("="*70)
        print("\nHello! I'm your IT support specialist with access to a comprehensive")
        print("knowledge base. I can help with:")
        print("  • Network and connectivity issues")
        print("  • System performance and crashes")
        print("  • Hardware troubleshooting")
        print("  • Software configuration")
        print("  • Security and privacy settings")
        print("  • And much more!")
        print("\nType 'help' for commands or 'quit' to exit")
        print("-"*70 + "\n")
    
    def display_help(self):
        """Display help information"""
        print("\n📋 Available Commands:")
        print("  'help'     - Show this help message")
        print("  'clear'    - Clear conversation history")
        print("  'examples' - Show example queries")
        print("  'sources'  - Show knowledge base sources")
        print("  'quit'     - Exit the program")
        print()
    
    def display_examples(self):
        """Display example queries"""
        print("\n💡 Example Queries:")
        print("  • My Wi-Fi keeps disconnecting")
        print("  • Computer is running very slow")
        print("  • How to fix blue screen error")
        print("  • Printer won't connect")
        print("  • Can't hear any sound from speakers")
        print("  • Email not syncing in Outlook")
        print("  • How to recover deleted files")
        print()
    
    def display_sources(self):
        """Display knowledge base sources"""
        print("\n📚 Knowledge Base Sources:")
        print("  • Microsoft Documentation")
        print("  • Stack Overflow Solutions")
        print("  • IT Best Practices")
        print("  • Common Troubleshooting Guides")
        print("  • Hardware Manufacturer Guides")
        print("\nThe system uses hybrid retrieval (BM25 + Dense Embeddings)")
        print("with Reciprocal Rank Fusion for optimal results.")
        print()
    
    def run(self):
        """Run the CLI interface"""
        self.display_welcome()
        
        while self.running:
            try:
                user_input = input("\n💬 You: ").strip()
                
                if not user_input:
                    continue
                
                # Handle commands
                if user_input.lower() == 'quit':
                    print("\n👋 Thank you for using Tech Support Assistant. Goodbye!")
                    self.running = False
                    break
                elif user_input.lower() == 'help':
                    self.display_help()
                    continue
                elif user_input.lower() == 'clear':
                    self.agent.reset_conversation()
                    print("✅ Conversation history cleared\n")
                    continue
                elif user_input.lower() == 'examples':
                    self.display_examples()
                    continue
                elif user_input.lower() == 'sources':
                    self.display_sources()
                    continue
                
                # Process the query
                print("\n🔍 Searching knowledge base and generating solution...\n")
                result = self.agent.generate_response(user_input)
                
                # Display response
                print("🤖 Assistant:", result['response'])
                print(f"\n[Category: {result['category']} | Context: {'✓' if result['has_context'] else '✗'}]")
                print("-"*70)
                
            except KeyboardInterrupt:
                print("\n\n👋 Interrupted. Goodbye!")
                self.running = False
            except Exception as e:
                print(f"\n❌ Error: {str(e)}\n")

# Main execution
if __name__ == "__main__":
    import sys
    from dotenv import load_dotenv
    
    # Load environment variables
    load_dotenv()
    
    # Get Groq API key
    GROQ_API_KEY = os.getenv("GROQ_API_KEY")
    
    if not GROQ_API_KEY:
        print("⚠️  GROQ_API_KEY not found in environment variables!")
        print("\n📝 To get your Groq API key:")
        print("1. Visit: https://console.groq.com/keys")
        print("2. Sign up or log in")
        print("3. Create a new API key")
        print("4. Set it as environment variable: export GROQ_API_KEY='your-key-here'")
        print("   Or add it to your .env file")
        sys.exit(1)
    
    # Available Groq models
    print("\n📊 Available Groq Models:")
    print("1. mixtral-8x7b-32768 (Recommended - Best performance)")
    print("2. llama2-70b-4096")
    print("3. gemma-7b-it")
    
    model_choice = input("\nSelect model (1-3, default=1): ").strip()
    
    model_map = {
        "1": "mixtral-8x7b-32768",
        "2": "llama2-70b-4096",
        "3": "gemma-7b-it"
    }
    
    model_name = model_map.get(model_choice, "mixtral-8x7b-32768")
    
    try:
        # Initialize agent
        print(f"\n🚀 Initializing Tech Support Agent with {model_name}...")
        agent = TechSupportAgent(GROQ_API_KEY, model_name)
        
        # Run CLI interface
        cli = CLIInterface(agent)
        cli.run()
        
    except Exception as e:
        print(f"❌ Failed to initialize: {str(e)}")
        print("\nMake sure you have installed required packages:")
        print("  pip install groq sentence-transformers rank-bm25 faiss-cpu")
