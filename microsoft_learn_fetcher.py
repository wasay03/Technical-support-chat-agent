# microsoft_learn_fetcher.py
# Fetch real technical documentation from Microsoft Learn

import requests
import json
import time
import re
from typing import Dict, List, Any, Optional
from dataclasses import dataclass, asdict
import logging
from bs4 import BeautifulSoup
from urllib.parse import urljoin, urlparse
import hashlib
import os
from pathlib import Path

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

@dataclass
class MicrosoftDoc:
    """Structure for Microsoft Learn documentation"""
    id: str
    title: str
    content: str
    category: str
    keywords: List[str]
    solutions: List[str]
    url: str
    metadata: Dict[str, Any]

class MicrosoftLearnFetcher:
    """Fetch and parse documentation from Microsoft Learn"""
    
    def __init__(self, cache_dir: str = "./cache/microsoft_learn"):
        """
        Initialize the Microsoft Learn fetcher
        
        Args:
            cache_dir: Directory to cache fetched content
        """
        self.base_url = "https://learn.microsoft.com"
        self.api_base = "https://learn.microsoft.com/api"
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        
        # Headers to mimic browser request
        self.headers = {
            'User-Agent': 'Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36',
            'Accept': 'application/json, text/plain, */*',
            'Accept-Language': 'en-US,en;q=0.9',
            'Cache-Control': 'no-cache'
        }
        
        # Key troubleshooting URLs from Microsoft Learn
        self.troubleshooting_urls = {
            "network": [
                "/en-us/windows/client-management/troubleshoot-networking",
                "/en-us/troubleshoot/windows-client/networking/tcp-ip-connectivity-issues",
                "/en-us/windows-server/networking/technologies/network-connectivity-status-indicator",
                "/en-us/windows/client-management/troubleshoot-tcpip-connectivity",
                "/en-us/troubleshoot/windows-client/networking/fix-network-connection-issues"
            ],
            "performance": [
                "/en-us/troubleshoot/windows-client/performance/windows-performance-issues-diagnostics",
                "/en-us/windows/client-management/troubleshoot-performance",
                "/en-us/troubleshoot/windows-client/performance/slow-performance-issues",
                "/en-us/windows-hardware/test/wpt/windows-performance-toolkit",
                "/en-us/troubleshoot/windows-client/performance/troubleshoot-slow-computer"
            ],
            "updates": [
                "/en-us/windows/deployment/update/windows-update-troubleshooting",
                "/en-us/troubleshoot/windows-client/installing-updates-features-roles/windows-update-issues-troubleshooting",
                "/en-us/windows/deployment/update/windows-update-errors",
                "/en-us/troubleshoot/windows-client/installing-updates-features-roles/common-windows-update-errors"
            ],
            "printing": [
                "/en-us/troubleshoot/windows-client/printing/printer-configuration-issues",
                "/en-us/windows-server/administration/windows-commands/print-command-reference",
                "/en-us/troubleshoot/windows-client/printing/troubleshoot-printing-issues",
                "/en-us/windows-hardware/drivers/print/troubleshooting-printer-drivers"
            ],
            "audio": [
                "/en-us/windows-hardware/drivers/audio/troubleshooting-audio",
                "/en-us/troubleshoot/windows-client/multimedia/no-sound-from-speakers",
                "/en-us/windows/client-management/troubleshoot-audio",
                "/en-us/troubleshoot/windows-client/multimedia/audio-playback-issues"
            ],
            "display": [
                "/en-us/troubleshoot/windows-client/display/troubleshoot-display-issues",
                "/en-us/windows-hardware/drivers/display/troubleshooting-display-drivers",
                "/en-us/troubleshoot/windows-client/display/screen-resolution-issues",
                "/en-us/windows/client-management/troubleshoot-display-problems"
            ],
            "security": [
                "/en-us/windows/security/threat-protection/windows-defender-antivirus/troubleshoot-windows-defender-antivirus",
                "/en-us/troubleshoot/windows-client/windows-security/windows-security-troubleshooting",
                "/en-us/windows/security/threat-protection/microsoft-defender-atp/troubleshoot-microsoft-defender-atp",
                "/en-us/windows/security/threat-protection/windows-firewall/troubleshooting-windows-firewall"
            ],
            "startup": [
                "/en-us/troubleshoot/windows-client/performance/troubleshoot-startup-problems",
                "/en-us/windows-hardware/manufacture/desktop/troubleshooting-windows-setup",
                "/en-us/troubleshoot/windows-client/performance/computer-startup-issues",
                "/en-us/windows/deployment/windows-autopilot/troubleshooting"
            ]
        }
        
        # Microsoft Learn API endpoints for search
        self.search_endpoint = f"{self.api_base}/search"
        self.content_endpoint = f"{self.api_base}/hierarchy"
    
    def _get_cache_key(self, url: str) -> str:
        """Generate cache key for URL"""
        return hashlib.md5(url.encode()).hexdigest()
    
    def _load_from_cache(self, url: str) -> Optional[Dict]:
        """Load cached content if available"""
        cache_key = self._get_cache_key(url)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        if cache_file.exists():
            try:
                with open(cache_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    # Check if cache is recent (within 7 days)
                    if time.time() - data.get('timestamp', 0) < 7 * 24 * 3600:
                        logger.info(f"Loaded from cache: {url}")
                        return data
            except Exception as e:
                logger.error(f"Error loading cache: {e}")
        
        return None
    
    def _save_to_cache(self, url: str, data: Dict):
        """Save content to cache"""
        cache_key = self._get_cache_key(url)
        cache_file = self.cache_dir / f"{cache_key}.json"
        
        data['timestamp'] = time.time()
        data['url'] = url
        
        try:
            with open(cache_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, ensure_ascii=False, indent=2)
            logger.info(f"Saved to cache: {url}")
        except Exception as e:
            logger.error(f"Error saving cache: {e}")
    
    def search_microsoft_learn(self, query: str, limit: int = 10) -> List[Dict]:
        """
        Search Microsoft Learn using their search API
        
        Args:
            query: Search query
            limit: Maximum number of results
        
        Returns:
            List of search results
        """
        try:
            params = {
                'search': query,
                'locale': 'en-us',
                '$top': limit,
                'facet': 'products',
                'category': 'Troubleshooting'
            }
            
            response = requests.get(
                self.search_endpoint,
                params=params,
                headers=self.headers,
                timeout=10
            )
            
            if response.status_code == 200:
                data = response.json()
                return data.get('results', [])
            else:
                logger.error(f"Search API error: {response.status_code}")
                return []
                
        except Exception as e:
            logger.error(f"Error searching Microsoft Learn: {e}")
            return []
    
    def fetch_page_content(self, url: str) -> Optional[Dict]:
        """
        Fetch and parse content from a Microsoft Learn page
        
        Args:
            url: Page URL (relative or absolute)
        
        Returns:
            Parsed content dictionary
        """
        # Convert to absolute URL if needed
        if not url.startswith('http'):
            url = urljoin(self.base_url, url)
        
        # Check cache first
        cached = self._load_from_cache(url)
        if cached:
            return cached
        
        try:
            response = requests.get(url, headers=self.headers, timeout=15)
            response.raise_for_status()
            
            soup = BeautifulSoup(response.text, 'html.parser')
            
            # Extract title
            title = ""
            title_elem = soup.find('h1') or soup.find('title')
            if title_elem:
                title = title_elem.get_text(strip=True)
            
            # Extract main content
            content = ""
            main_content = soup.find('main') or soup.find('article')
            if main_content:
                # Remove script and style elements
                for script in main_content(['script', 'style']):
                    script.decompose()
                content = main_content.get_text(separator='\n', strip=True)
            
            # Extract steps/solutions
            solutions = []
            
            # Look for numbered lists or procedure sections
            for ol in soup.find_all('ol'):
                for li in ol.find_all('li'):
                    step_text = li.get_text(strip=True)
                    if step_text and len(step_text) > 10:
                        solutions.append(step_text)
            
            # Look for resolution sections
            for section in soup.find_all(['section', 'div']):
                header = section.find(['h2', 'h3', 'h4'])
                if header and any(word in header.get_text().lower() 
                                 for word in ['resolution', 'solution', 'fix', 'troubleshoot', 'steps']):
                    steps = section.find_all('li')
                    for step in steps:
                        step_text = step.get_text(strip=True)
                        if step_text and step_text not in solutions:
                            solutions.append(step_text)
            
            # Extract keywords from meta tags
            keywords = []
            meta_keywords = soup.find('meta', {'name': 'keywords'})
            if meta_keywords:
                keywords = [k.strip() for k in meta_keywords.get('content', '').split(',')]
            
            # Also extract from breadcrumbs
            breadcrumbs = soup.find('nav', {'aria-label': 'Breadcrumb'})
            if breadcrumbs:
                for crumb in breadcrumbs.find_all('a'):
                    keyword = crumb.get_text(strip=True).lower()
                    if keyword and keyword not in keywords:
                        keywords.append(keyword)
            
            # Determine category
            category = self._determine_category(url, title, content)
            
            result = {
                'title': title,
                'content': content[:2000],  # Limit content length
                'solutions': solutions[:10],  # Limit to top 10 solutions
                'keywords': keywords[:20],  # Limit keywords
                'category': category,
                'url': url,
                'source': 'Microsoft Learn'
            }
            
            # Save to cache
            self._save_to_cache(url, result)
            
            # Be polite to the server
            time.sleep(0.5)
            
            return result
            
        except Exception as e:
            logger.error(f"Error fetching {url}: {e}")
            return None
    
    def _determine_category(self, url: str, title: str, content: str) -> str:
        """Determine the category based on URL and content"""
        url_lower = url.lower()
        combined = f"{url_lower} {title.lower()} {content[:500].lower()}"
        
        if any(word in combined for word in ['network', 'wifi', 'ethernet', 'tcp', 'ip', 'dns']):
            return "Network"
        elif any(word in combined for word in ['performance', 'slow', 'memory', 'cpu', 'disk']):
            return "System"
        elif any(word in combined for word in ['update', 'patch', 'upgrade', 'installation']):
            return "System"
        elif any(word in combined for word in ['print', 'printer', 'spool']):
            return "Hardware"
        elif any(word in combined for word in ['audio', 'sound', 'speaker', 'microphone']):
            return "Hardware"
        elif any(word in combined for word in ['display', 'screen', 'monitor', 'graphics']):
            return "Hardware"
        elif any(word in combined for word in ['security', 'antivirus', 'firewall', 'defender']):
            return "Security"
        elif any(word in combined for word in ['boot', 'startup', 'bios', 'uefi']):
            return "System"
        else:
            return "General"
    
    def fetch_troubleshooting_docs(self) -> Dict[str, Any]:
        """
        Fetch comprehensive troubleshooting documentation from Microsoft Learn
        
        Returns:
            Dictionary of categorized troubleshooting documents
        """
        all_docs = {}
        
        logger.info("Fetching Microsoft Learn troubleshooting documentation...")
        
        for category, urls in self.troubleshooting_urls.items():
            logger.info(f"Fetching {category} documentation...")
            
            for url in urls:
                try:
                    doc_data = self.fetch_page_content(url)
                    
                    if doc_data and doc_data.get('content'):
                        # Generate unique ID
                        doc_id = f"ms_learn_{category}_{self._get_cache_key(url)[:8]}"
                        
                        # Format for the knowledge base
                        all_docs[doc_id] = {
                            'title': doc_data['title'],
                            'content': doc_data['content'],
                            'category': doc_data['category'],
                            'keywords': doc_data['keywords'],
                            'solutions': doc_data['solutions'],
                            'metadata': {
                                'source': 'Microsoft Learn',
                                'url': doc_data['url'],
                                'fetched_at': time.time()
                            }
                        }
                        
                        logger.info(f"✓ Fetched: {doc_data['title']}")
                    
                except Exception as e:
                    logger.error(f"Error processing {url}: {e}")
                    continue
        
        # Also search for common issues
        common_issues = [
            "Windows 11 troubleshooting",
            "fix network connection Windows",
            "Windows update error",
            "printer not working Windows",
            "no sound Windows 11",
            "blue screen of death",
            "Windows startup repair",
            "Windows defender issues"
        ]
        
        logger.info("Searching for common issues...")
        
        for issue in common_issues:
            try:
                search_results = self.search_microsoft_learn(issue, limit=3)
                
                for result in search_results:
                    if 'url' in result:
                        doc_data = self.fetch_page_content(result['url'])
                        
                        if doc_data and doc_data.get('content'):
                            doc_id = f"ms_learn_search_{self._get_cache_key(result['url'])[:8]}"
                            
                            if doc_id not in all_docs:
                                all_docs[doc_id] = {
                                    'title': doc_data['title'],
                                    'content': doc_data['content'],
                                    'category': doc_data['category'],
                                    'keywords': doc_data['keywords'],
                                    'solutions': doc_data['solutions'],
                                    'metadata': {
                                        'source': 'Microsoft Learn',
                                        'url': doc_data['url'],
                                        'search_query': issue,
                                        'fetched_at': time.time()
                                    }
                                }
                                
                                logger.info(f"✓ Found: {doc_data['title']}")
                
            except Exception as e:
                logger.error(f"Error searching for {issue}: {e}")
                continue
        
        logger.info(f"Fetched {len(all_docs)} documents from Microsoft Learn")
        
        # Save all documents to a single file for reference
        output_file = self.cache_dir / "all_microsoft_docs.json"
        with open(output_file, 'w', encoding='utf-8') as f:
            json.dump(all_docs, f, ensure_ascii=False, indent=2)
        
        logger.info(f"Saved all documents to {output_file}")
        
        return all_docs
    
    def fetch_specific_topic(self, topic: str) -> List[Dict]:
        """
        Fetch documentation for a specific topic
        
        Args:
            topic: Topic to search for
        
        Returns:
            List of relevant documents
        """
        docs = []
        
        # Search for the topic
        search_results = self.search_microsoft_learn(topic, limit=5)
        
        for result in search_results:
            if 'url' in result:
                doc_data = self.fetch_page_content(result['url'])
                
                if doc_data:
                    docs.append(doc_data)
        
        return docs

class EnhancedKnowledgeBaseBuilder:
    """Enhanced knowledge base builder with Microsoft Learn integration"""
    
    @staticmethod
    def fetch_from_microsoft_docs() -> Dict[str, Any]:
        """
        Fetch real documentation from Microsoft Learn
        
        Returns:
            Dictionary of Microsoft Learn documents
        """
        fetcher = MicrosoftLearnFetcher()
        
        # Check if we have recent cached data
        cache_file = Path("./cache/microsoft_learn/all_microsoft_docs.json")
        
        if cache_file.exists():
            # Check if cache is recent (within 24 hours)
            if time.time() - cache_file.stat().st_mtime < 24 * 3600:
                logger.info("Using cached Microsoft Learn data")
                with open(cache_file, 'r', encoding='utf-8') as f:
                    return json.load(f)
        
        # Fetch fresh data
        logger.info("Fetching fresh data from Microsoft Learn...")
        return fetcher.fetch_troubleshooting_docs()
    
    @staticmethod
    def fetch_from_stackoverflow() -> Dict[str, Any]:
        """
        Fetch common IT solutions from Stack Overflow
        Note: In production, use Stack Exchange API with proper authentication
        """
        # Stack Overflow API for Windows troubleshooting questions
        so_docs = {}
        
        try:
            # Example using Stack Exchange API (requires registration for higher limits)
            base_url = "https://api.stackexchange.com/2.3/questions"
            
            # Common Windows troubleshooting tags
            tags = [
                "windows-10", "windows-11", "networking", "wifi",
                "printer", "audio", "display", "bsod", "performance"
            ]
            
            for tag in tags:
                params = {
                    'order': 'desc',
                    'sort': 'votes',
                    'tagged': tag,
                    'site': 'stackoverflow',
                    'filter': 'withbody',
                    'pagesize': 5
                }
                
                try:
                    response = requests.get(base_url, params=params)
                    if response.status_code == 200:
                        data = response.json()
                        
                        for item in data.get('items', []):
                            if item.get('is_answered'):
                                doc_id = f"so_{tag}_{item['question_id']}"
                                
                                # Extract solutions from accepted answer
                                solutions = []
                                if 'accepted_answer_id' in item:
                                    # Note: Would need additional API call for answer body
                                    solutions.append(f"View solution at: {item['link']}")
                                
                                so_docs[doc_id] = {
                                    'title': item['title'],
                                    'content': BeautifulSoup(item.get('body', ''), 'html.parser').get_text()[:1000],
                                    'category': 'General',
                                    'keywords': item.get('tags', []),
                                    'solutions': solutions,
                                    'metadata': {
                                        'source': 'Stack Overflow',
                                        'url': item['link'],
                                        'score': item.get('score', 0),
                                        'view_count': item.get('view_count', 0)
                                    }
                                }
                    
                    # Be polite to the API
                    time.sleep(1)
                    
                except Exception as e:
                    logger.error(f"Error fetching Stack Overflow data for {tag}: {e}")
                    continue
            
        except Exception as e:
            logger.error(f"Error accessing Stack Overflow API: {e}")
        
        # Fallback to curated common solutions if API fails
        if not so_docs:
            so_docs = {
                "so_network_1": {
                    "title": "Windows Network Adapter Reset",
                    "content": "How to completely reset network adapters in Windows when experiencing connectivity issues.",
                    "category": "Network",
                    "keywords": ["network", "adapter", "reset", "tcp/ip", "winsock"],
                    "solutions": [
                        "Open Command Prompt as Administrator",
                        "Run: netsh winsock reset",
                        "Run: netsh int ip reset",
                        "Run: ipconfig /release",
                        "Run: ipconfig /renew",
                        "Run: ipconfig /flushdns",
                        "Restart your computer"
                    ],
                    "metadata": {"source": "Stack Overflow Community"}
                }
            }
        
        return so_docs
    
    @staticmethod
    def build_complete_kb() -> List[Any]:
        """
        Build complete knowledge base from all sources
        
        Returns:
            List of Document objects
        """
        from tech_support_agent import Document
        
        kb = {}
        
        # Fetch from Microsoft Learn
        logger.info("Building knowledge base from Microsoft Learn...")
        kb.update(EnhancedKnowledgeBaseBuilder.fetch_from_microsoft_docs())
        
        # Fetch from Stack Overflow
        logger.info("Adding Stack Overflow solutions...")
        kb.update(EnhancedKnowledgeBaseBuilder.fetch_from_stackoverflow())
        
        # Convert to Document objects
        documents = []
        for doc_id, data in kb.items():
            doc = Document(
                id=doc_id,
                title=data.get("title", ""),
                content=data.get("content", ""),
                category=data.get("category", "General"),
                keywords=data.get("keywords", []),
                solutions=data.get("solutions", []),
                metadata=data.get("metadata", {})
            )
            documents.append(doc)
        
        logger.info(f"Built knowledge base with {len(documents)} documents")
        
        return documents

# Example usage
if __name__ == "__main__":
    # Test the fetcher
    print("Testing Microsoft Learn Fetcher...")
    
    # Initialize fetcher
    fetcher = MicrosoftLearnFetcher()
    
    # Test search
    print("\n1. Testing search functionality...")
    results = fetcher.search_microsoft_learn("Windows network troubleshooting", limit=3)
    for result in results:
        print(f"  - {result.get('title', 'No title')}")
    
    # Test fetching a specific page
    print("\n2. Testing page fetching...")
    test_url = "/en-us/windows/client-management/troubleshoot-networking"
    content = fetcher.fetch_page_content(test_url)
    if content:
        print(f"  ✓ Fetched: {content['title']}")
        print(f"    Solutions found: {len(content.get('solutions', []))}")
    
    # Test building complete knowledge base
    print("\n3. Building complete knowledge base...")
    kb = EnhancedKnowledgeBaseBuilder.build_complete_kb()
    print(f"  ✓ Built KB with {len(kb)} documents")
    
    # Show categories
    categories = {}
    for doc in kb:
        cat = doc.category
        categories[cat] = categories.get(cat, 0) + 1
    
    print("\n4. Knowledge Base Categories:")
    for cat, count in sorted(categories.items()):
        print(f"  - {cat}: {count} documents")
    
    print("\n✅ Testing complete!")