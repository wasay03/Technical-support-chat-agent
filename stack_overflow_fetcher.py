# Stack Overflow Data Fetcher using Stack Exchange API
# Enhanced with error handling and rate limiting

import os
import json
import time
import logging
from typing import Dict, List, Optional, Any
import requests
from datetime import datetime, timedelta
from dataclasses import dataclass
from enum import Enum

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class StackExchangeAPI:
    """Stack Exchange API client for fetching technical support data"""
    
    BASE_URL = "https://api.stackexchange.com/2.3"
    
    def __init__(self, api_key: Optional[str] = None):
        """
        Initialize Stack Exchange API client
        
        Args:
            api_key: Optional API key for higher rate limits
        """
        self.api_key = api_key
        self.session = requests.Session()
        self.session.headers.update({
            'User-Agent': 'TechSupportAgent/1.0 (https://github.com/your-repo)'
        })
        
        # Rate limiting
        self.last_request_time = 0
        self.min_request_interval = 0.1  # 100ms between requests
    
    def _rate_limit(self):
        """Implement rate limiting"""
        current_time = time.time()
        time_since_last = current_time - self.last_request_time
        
        if time_since_last < self.min_request_interval:
            time.sleep(self.min_request_interval - time_since_last)
        
        self.last_request_time = time.time()
    
    def _make_request(self, endpoint: str, params: Dict[str, Any]) -> Optional[Dict]:
        """Make API request with error handling"""
        self._rate_limit()
        
        try:
            # Add API key if available
            if self.api_key:
                params['key'] = self.api_key
            
            # Add site parameter for Stack Overflow
            params['site'] = 'stackoverflow'
            
            url = f"{self.BASE_URL}/{endpoint}"
            response = self.session.get(url, params=params)
            
            if response.status_code == 200:
                return response.json()
            elif response.status_code == 429:
                logger.warning("Rate limit exceeded, waiting...")
                time.sleep(60)  # Wait 1 minute
                return self._make_request(endpoint, params)
            else:
                logger.error(f"API request failed: {response.status_code} - {response.text}")
                return None
                
        except Exception as e:
            logger.error(f"Request error: {str(e)}")
            return None

class StackOverflowFetcher:
    """Fetch technical support data from Stack Overflow"""
    
    def __init__(self, api_key: Optional[str] = None):
        self.api = StackExchangeAPI(api_key)
        
        # Common technical support tags
        self.tech_support_tags = [
            "windows", "macos", "linux", "networking", "hardware", "software",
            "printer", "audio", "display", "browser", "email", "security",
            "performance", "crash", "bsod", "wifi", "internet", "driver",
            "update", "installation", "configuration", "troubleshooting"
        ]
    
    def fetch_questions_by_tag(self, tag: str, days_back: int = 30, max_questions: int = 50) -> List[Dict]:
        """
        Fetch questions by tag with filtering
        
        Args:
            tag: Tag to search for
            days_back: Number of days to look back
            max_questions: Maximum number of questions to fetch
        
        Returns:
            List of question data
        """
        from_date = int((datetime.now() - timedelta(days=days_back)).timestamp())
        
        params = {
            'fromdate': from_date,
            'pagesize': min(max_questions, 100),  # API max is 100
            'order': 'desc',
            'sort': 'votes',
            'tagged': tag,
            'filter': 'withbody',  # Include question body
            'accepted': 'true'  # Only accepted answers
        }
        
        data = self.api._make_request('questions', params)
        
        if not data or 'items' not in data:
            logger.warning(f"No data returned for tag: {tag}")
            return []
        
        return data['items']
    
    def fetch_answers_for_question(self, question_id: int) -> List[Dict]:
        """Fetch answers for a specific question"""
        params = {
            'order': 'desc',
            'sort': 'votes',
            'filter': 'withbody'
        }
        
        data = self.api._make_request(f'questions/{question_id}/answers', params)
        
        if not data or 'items' not in data:
            return []
        
        return data['items']
    
    def extract_solutions_from_answers(self, answers: List[Dict]) -> List[str]:
        """Extract solution steps from answer content"""
        solutions = []
        
        for answer in answers:
            if answer.get('is_accepted', False) and answer.get('score', 0) > 0:
                content = answer.get('body', '')
                
                # Extract code blocks and numbered lists
                import re
                
                # Find code blocks
                code_blocks = re.findall(r'<code>(.*?)</code>', content, re.DOTALL)
                for code in code_blocks:
                    if len(code.strip()) > 10:  # Minimum length for meaningful code
                        solutions.append(f"Code solution: {code.strip()}")
                
                # Find numbered lists
                numbered_items = re.findall(r'<li>(.*?)</li>', content)
                for item in numbered_items:
                    if len(item.strip()) > 20:  # Minimum length for meaningful step
                        solutions.append(item.strip())
                
                # Extract plain text steps
                # Remove HTML tags
                import html
                clean_text = html.unescape(content)
                clean_text = re.sub(r'<[^>]+>', '', clean_text)
                
                # Split into sentences and find potential steps
                sentences = re.split(r'[.!?]+', clean_text)
                for sentence in sentences:
                    sentence = sentence.strip()
                    if (len(sentence) > 30 and 
                        any(keyword in sentence.lower() for keyword in 
                            ['step', 'first', 'then', 'next', 'finally', 'check', 'verify', 'run', 'open'])):
                        solutions.append(sentence)
        
        return solutions[:10]  # Limit to 10 solutions
    
    def categorize_question(self, question: Dict) -> str:
        """Categorize question based on tags and content"""
        tags = question.get('tags', [])
        title = question.get('title', '').lower()
        
        # Category mapping
        category_keywords = {
            'Network': ['wifi', 'internet', 'network', 'connection', 'ethernet', 'router'],
            'System': ['windows', 'macos', 'linux', 'bsod', 'crash', 'performance', 'slow'],
            'Hardware': ['printer', 'audio', 'speaker', 'monitor', 'display', 'keyboard', 'mouse'],
            'Software': ['browser', 'email', 'outlook', 'application', 'program'],
            'Security': ['virus', 'malware', 'firewall', 'password', 'security'],
            'Settings': ['settings', 'configuration', 'setup', 'install']
        }
        
        # Check tags and title for category
        for category, keywords in category_keywords.items():
            if any(kw in title for kw in keywords) or any(kw in tags for kw in keywords):
                return category
        
        return 'General'
    
    def fetch_tech_support_data(self, max_questions_per_tag: int = 20) -> Dict[str, Dict]:
        """
        Fetch comprehensive technical support data from Stack Overflow
        
        Returns:
            Dictionary of technical support topics with solutions
        """
        logger.info("Fetching technical support data from Stack Overflow...")
        
        tech_support_data = {}
        
        for tag in self.tech_support_tags:
            logger.info(f"Fetching questions for tag: {tag}")
            
            questions = self.fetch_questions_by_tag(tag, days_back=90, max_questions=max_questions_per_tag)
            
            for question in questions:
                if question.get('score', 0) < 1:  # Skip low-quality questions
                    continue
                
                # Get answers for this question
                answers = self.fetch_answers_for_question(question['question_id'])
                
                if not answers:
                    continue
                
                # Extract solutions
                solutions = self.extract_solutions_from_answers(answers)
                
                if not solutions:
                    continue
                
                # Create topic entry
                topic_id = f"so_{tag}_{question['question_id']}"
                category = self.categorize_question(question)
                
                # Extract keywords from tags and title
                keywords = question.get('tags', [])
                title_keywords = question.get('title', '').lower().split()
                keywords.extend([kw for kw in title_keywords if len(kw) > 3])
                
                tech_support_data[topic_id] = {
                    "title": question.get('title', ''),
                    "content": f"Stack Overflow question: {question.get('title', '')}",
                    "category": category,
                    "keywords": list(set(keywords)),  # Remove duplicates
                    "solutions": solutions,
                    "metadata": {
                        "source": "stackoverflow",
                        "question_id": question['question_id'],
                        "score": question.get('score', 0),
                        "answer_count": question.get('answer_count', 0),
                        "created_date": question.get('creation_date', 0),
                        "tags": question.get('tags', [])
                    }
                }
                
                # Add delay to avoid overwhelming the API
                time.sleep(0.5)
        
        logger.info(f"Fetched {len(tech_support_data)} technical support topics from Stack Overflow")
        return tech_support_data
    
    def get_fallback_data(self) -> Dict[str, Dict]:
        """Return fallback data if API is unavailable"""
        return {
            "so_bsod_fallback": {
                "title": "Blue Screen of Death (BSOD) Resolution",
                "content": "Common solutions for Windows blue screen errors from Stack Overflow community",
                "category": "System",
                "keywords": ["bsod", "blue screen", "crash", "error", "stop code", "windows"],
                "solutions": [
                    "Note the stop code and error message displayed",
                    "Boot into Safe Mode if possible",
                    "Check for recent hardware or software changes",
                    "Run memory diagnostic: mdsched.exe",
                    "Update or roll back drivers",
                    "Check system files: sfc /scannow",
                    "Review dump files with BlueScreenView",
                    "Test RAM with MemTest86",
                    "Check hard drive health",
                    "Perform system restore or reset if needed"
                ],
                "metadata": {
                    "source": "stackoverflow_fallback",
                    "tags": ["windows", "bsod", "crash"]
                }
            },
            "so_network_fallback": {
                "title": "Network Connectivity Issues",
                "content": "Common network troubleshooting solutions from Stack Overflow",
                "category": "Network",
                "keywords": ["wifi", "network", "connection", "internet", "ethernet"],
                "solutions": [
                    "Check physical connections and cables",
                    "Verify Wi-Fi is enabled and airplane mode is off",
                    "Run Windows Network Troubleshooter",
                    "Reset network adapter: netsh winsock reset",
                    "Update network drivers from Device Manager",
                    "Flush DNS cache: ipconfig /flushdns",
                    "Check router status and restart if needed",
                    "Verify DHCP is enabled or static IP is correct",
                    "Disable VPN or proxy if connected",
                    "Check Windows Firewall settings"
                ],
                "metadata": {
                    "source": "stackoverflow_fallback",
                    "tags": ["networking", "wifi", "connection"]
                }
            }
        }

def fetch_from_stackoverflow(api_key: Optional[str] = None) -> Dict[str, Dict]:
    """
    Main function to fetch Stack Overflow data
    
    Args:
        api_key: Optional Stack Exchange API key for higher rate limits
    
    Returns:
        Dictionary of technical support topics
    """
    try:
        fetcher = StackOverflowFetcher(api_key)
        return fetcher.fetch_tech_support_data()
    except Exception as e:
        logger.error(f"Error fetching from Stack Overflow: {str(e)}")
        logger.info("Using fallback data...")
        fetcher = StackOverflowFetcher()
        return fetcher.get_fallback_data()

if __name__ == "__main__":
    # Test the fetcher
    import os
    from dotenv import load_dotenv
    
    load_dotenv()
    
    # Get API key from environment
    api_key = os.getenv("STACK_EXCHANGE_API_KEY")
    
    print("Testing Stack Overflow Fetcher...")
    data = fetch_from_stackoverflow(api_key)
    
    print(f"\nFetched {len(data)} topics:")
    for topic_id, topic_data in data.items():
        print(f"- {topic_data['title']} ({topic_data['category']})")
        print(f"  Solutions: {len(topic_data['solutions'])}")
        print() 