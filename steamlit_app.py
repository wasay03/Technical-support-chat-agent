# streamlit_app.py
# Enhanced Web Interface for Technical Support Chat Agent with Groq and Hybrid Retrieval

import streamlit as st
import os
from datetime import datetime
import json
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import pandas as pd
from tech_support_agent import (
    TechSupportAgent, 
    IssueCategory, 
    KnowledgeBaseBuilder,
    HybridRetriever
)

# Page configuration
st.set_page_config(
    page_title="Tech Support AI - Powered by Groq",
    page_icon="🤖",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Custom CSS for better UI
st.markdown("""
    <style>
    .stApp {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
    }
    .main > div {
        background-color: rgba(255, 255, 255, 0.95);
        border-radius: 10px;
        padding: 2rem;
        margin: 1rem;
    }
    .chat-message {
        padding: 1.5rem;
        border-radius: 10px;
        margin-bottom: 1rem;
        animation: fadeIn 0.5s;
    }
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    .user-message {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        margin-left: 20%;
    }
    .assistant-message {
        background: white;
        border: 2px solid #e0e0e0;
        margin-right: 20%;
    }
    .metric-card {
        background: white;
        padding: 1rem;
        border-radius: 10px;
        box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        text-align: center;
    }
    .category-badge {
        display: inline-block;
        padding: 0.35rem 0.65rem;
        border-radius: 20px;
        font-size: 0.875rem;
        font-weight: 600;
        margin-top: 0.5rem;
    }
    .network-badge { background: #e3f2fd; color: #1565c0; }
    .system-badge { background: #e8f5e9; color: #2e7d32; }
    .hardware-badge { background: #fff3e0; color: #e65100; }
    .software-badge { background: #f3e5f5; color: #6a1b9a; }
    .security-badge { background: #ffebee; color: #c62828; }
    .settings-badge { background: #e0f2f1; color: #00695c; }
    .general-badge { background: #f5f5f5; color: #424242; }
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white;
        border: none;
        padding: 0.5rem 1rem;
        border-radius: 5px;
        font-weight: 600;
        transition: transform 0.2s;
    }
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 4px 8px rgba(0,0,0,0.2);
    }
    </style>
    """, unsafe_allow_html=True)

# Initialize session state
if 'agent' not in st.session_state:
    st.session_state.agent = None
    st.session_state.messages = []
    st.session_state.initialized = False
    st.session_state.stats = {
        'total_queries': 0,
        'categories': {},
        'response_times': [],
        'context_usage': {'with_context': 0, 'without_context': 0}
    }

# Sidebar configuration
with st.sidebar:
    st.title("⚙️ Configuration")
    
    # Groq API Configuration
    st.markdown("### 🚀 Groq API Setup")
    
    groq_api_key = st.text_input(
        "Groq API Key",
        type="password",
        placeholder="gsk_...",
        help="Get your API key from https://console.groq.com/keys"
    )
    
    # Model selection with descriptions
    model_info = {
        "mixtral-8x7b-32768": "**Mixtral 8x7B** - Best overall performance, 32k context",
        "llama2-70b-4096": "**Llama 2 70B** - Powerful open model, 4k context",
        "gemma-7b-it": "**Gemma 7B** - Fast and efficient, good for quick responses"
    }
    
    selected_model = st.selectbox(
        "Select Model",
        options=list(model_info.keys()),
        format_func=lambda x: x.split('-')[0].title(),
        help="Choose the Groq model for responses"
    )
    
    st.markdown(model_info[selected_model])
    
    # Retrieval settings
    st.markdown("### 🔍 Retrieval Settings")
    
    num_results = st.slider(
        "Number of KB results",
        min_value=1,
        max_value=5,
        value=3,
        help="Number of knowledge base articles to retrieve"
    )
    
    hybrid_alpha = st.slider(
        "RRF Alpha",
        min_value=1,
        max_value=100,
        value=60,
        help="Reciprocal Rank Fusion parameter (higher = more weight on top ranks)"
    )
    
    # Initialize button
    if st.button("🎯 Initialize Agent", type="primary", use_container_width=True):
        if not groq_api_key:
            st.error("Please enter your Groq API key!")
            st.markdown("[Get your API key here](https://console.groq.com/keys)")
        else:
            try:
                with st.spinner("🔄 Building knowledge base and initializing agent..."):
                    st.session_state.agent = TechSupportAgent(groq_api_key, selected_model)
                    st.session_state.initialized = True
                    st.success("✅ Agent initialized successfully!")
                    st.balloons()
            except Exception as e:
                st.error(f"Failed to initialize: {str(e)}")
    
    if st.session_state.initialized:
        st.divider()
        
        # Knowledge Base Info
        st.markdown("### 📚 Knowledge Base")
        
        if st.session_state.agent:
            kb_size = len(st.session_state.agent.retriever.documents)
            st.metric("Documents", kb_size)
            
            # Show categories
            categories = {}
            for doc in st.session_state.agent.retriever.documents:
                cat = doc.category
                categories[cat] = categories.get(cat, 0) + 1
            
            st.markdown("**Categories:**")
            for cat, count in sorted(categories.items()):
                st.write(f"• {cat}: {count} docs")
        
        st.divider()
        
        # Quick Actions
        st.markdown("### ⚡ Quick Actions")
        
        col1, col2 = st.columns(2)
        
        with col1:
            if st.button("🗑️ Clear Chat", use_container_width=True):
                st.session_state.messages = []
                if st.session_state.agent:
                    st.session_state.agent.reset_conversation()
                st.rerun()
        
        with col2:
            if st.button("📊 Show Stats", use_container_width=True):
                st.session_state.show_stats = True
        
        # Export conversation
        if st.button("💾 Export Chat", use_container_width=True):
            if st.session_state.messages:
                conversation_data = {
                    "timestamp": datetime.now().isoformat(),
                    "model": selected_model,
                    "messages": st.session_state.messages,
                    "stats": st.session_state.stats
                }
                st.download_button(
                    label="📥 Download JSON",
                    data=json.dumps(conversation_data, indent=2),
                    file_name=f"tech_support_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
                    mime="application/json"
                )

# Main content area
st.title("🤖 Advanced Tech Support AI")
st.markdown("### Powered by Groq with Hybrid RAG (BM25 + Dense Retrieval)")

if not st.session_state.initialized:
    # Welcome screen
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.markdown("""
        <div class="metric-card">
        <h3>🚀 Fast Responses</h3>
        <p>Powered by Groq's LPU for ultra-fast inference</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="metric-card">
        <h3>📚 Comprehensive KB</h3>
        <p>Access to extensive technical documentation</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="metric-card">
        <h3>🎯 Hybrid Search</h3>
        <p>BM25 + Dense embeddings for best results</p>
        </div>
        """, unsafe_allow_html=True)
    
    st.info("👈 Configure and initialize the agent in the sidebar to start")
    
    # Show example queries
    with st.expander("💡 Example Queries", expanded=True):
        examples = [
            "My Wi-Fi keeps disconnecting every few minutes",
            "Computer shows blue screen with error code 0x0000001E",
            "Outlook won't sync my emails",
            "How to recover files deleted from Recycle Bin",
            "Printer says 'offline' but it's connected",
            "No sound from speakers after Windows update",
            "Computer is extremely slow after startup",
            "How to set up Remote Desktop connection"
        ]
        
        cols = st.columns(2)
        for i, example in enumerate(examples):
            with cols[i % 2]:
                if st.button(example, key=f"example_{i}", use_container_width=True):
                    st.session_state.example_query = example

else:
    # Chat interface
    tabs = st.tabs(["💬 Chat", "📊 Analytics", "📚 Knowledge Base"])
    
    with tabs[0]:
        # Display conversation history
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.write(message["content"])
                if "category" in message and message["role"] == "assistant":
                    category = message["category"].lower().replace(" ", "-")
                    st.markdown(
                        f'<span class="category-badge {category}-badge">{message["category"]}</span>',
                        unsafe_allow_html=True
                    )
                    if "has_context" in message:
                        context_status = "✓ Using KB" if message["has_context"] else "✗ No KB"
                        st.caption(f"{context_status} | {message.get('timestamp', '')}")
        
        # Chat input
        if 'example_query' in st.session_state:
            prompt = st.session_state.example_query
            del st.session_state.example_query
        else:
            prompt = st.chat_input("Describe your technical issue...")
        
        if prompt:
            # Add user message
            st.session_state.messages.append({
                "role": "user",
                "content": prompt,
                "timestamp": datetime.now().isoformat()
            })
            
            # Display user message
            with st.chat_message("user"):
                st.write(prompt)
            
            # Generate response
            with st.chat_message("assistant"):
                with st.spinner("🔍 Searching knowledge base and generating solution..."):
                    try:
                        import time
                        start_time = time.time()
                        
                        result = st.session_state.agent.generate_response(prompt)
                        response_time = time.time() - start_time
                        
                        st.write(result["response"])
                        
                        # Display metadata
                        category = result["category"]
                        category_lower = category.lower().replace(" ", "-")
                        st.markdown(
                            f'<span class="category-badge {category_lower}-badge">{category}</span>',
                            unsafe_allow_html=True
                        )
                        
                        context_status = "✓ Using KB" if result.get("has_context") else "✗ No KB"
                        st.caption(f"{context_status} | Response time: {response_time:.2f}s")
                        
                        # Update statistics
                        st.session_state.stats['total_queries'] += 1
                        st.session_state.stats['categories'][category] = \
                            st.session_state.stats['categories'].get(category, 0) + 1
                        st.session_state.stats['response_times'].append(response_time)
                        if result.get("has_context"):
                            st.session_state.stats['context_usage']['with_context'] += 1
                        else:
                            st.session_state.stats['context_usage']['without_context'] += 1
                        
                        # Add to messages
                        st.session_state.messages.append({
                            "role": "assistant",
                            "content": result["response"],
                            "category": category,
                            "has_context": result.get("has_context"),
                            "timestamp": result["timestamp"]
                        })
                        
                    except Exception as e:
                        st.error(f"Error: {str(e)}")
        
        # Feedback buttons
        if st.session_state.messages:
            st.divider()
            col1, col2, col3 = st.columns(3)
            
            with col1:
                if st.button("👍 Helpful", use_container_width=True):
                    st.success("Thank you for your feedback!")
            
            with col2:
                if st.button("👎 Not Helpful", use_container_width=True):
                    st.info("Please try rephrasing your question with more details.")
            
            with col3:
                if st.button("📞 Human Support", use_container_width=True):
                    st.info("For complex issues, contact your IT department.")
    
    with tabs[1]:
        # Analytics Dashboard
        st.markdown("### 📊 Session Analytics")
        
        if st.session_state.stats['total_queries'] > 0:
            # Overview metrics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Queries", st.session_state.stats['total_queries'])
            
            with col2:
                avg_time = sum(st.session_state.stats['response_times']) / len(st.session_state.stats['response_times'])
                st.metric("Avg Response Time", f"{avg_time:.2f}s")
            
            with col3:
                kb_usage = st.session_state.stats['context_usage']['with_context']
                total = sum(st.session_state.stats['context_usage'].values())
                kb_percent = (kb_usage / total * 100) if total > 0 else 0
                st.metric("KB Usage", f"{kb_percent:.0f}%")
            
            with col4:
                categories_count = len(st.session_state.stats['categories'])
                st.metric("Issue Categories", categories_count)
            
            # Charts
            if st.session_state.stats['categories']:
                # Create subplots
                fig = make_subplots(
                    rows=2, cols=2,
                    subplot_titles=("Issue Categories", "Response Times", "KB Usage", "Category Timeline"),
                    specs=[[{"type": "pie"}, {"type": "box"}],
                           [{"type": "bar"}, {"type": "scatter"}]]
                )
                
                # Pie chart for categories
                categories = list(st.session_state.stats['categories'].keys())
                values = list(st.session_state.stats['categories'].values())
                fig.add_trace(
                    go.Pie(labels=categories, values=values, hole=0.3),
                    row=1, col=1
                )
                
                # Box plot for response times
                if st.session_state.stats['response_times']:
                    fig.add_trace(
                        go.Box(y=st.session_state.stats['response_times'], name="Response Time"),
                        row=1, col=2
                    )
                
                # Bar chart for KB usage
                kb_data = st.session_state.stats['context_usage']
                fig.add_trace(
                    go.Bar(
                        x=["With KB", "Without KB"],
                        y=[kb_data['with_context'], kb_data['without_context']],
                        marker_color=['green', 'gray']
                    ),
                    row=2, col=1
                )
                
                # Timeline scatter plot
                if len(st.session_state.messages) > 0:
                    timeline_data = []
                    for i, msg in enumerate(st.session_state.messages):
                        if msg["role"] == "assistant" and "category" in msg:
                            timeline_data.append({
                                'index': i,
                                'category': msg["category"]
                            })
                    
                    if timeline_data:
                        df = pd.DataFrame(timeline_data)
                        for cat in df['category'].unique():
                            cat_df = df[df['category'] == cat]
                            fig.add_trace(
                                go.Scatter(
                                    x=cat_df['index'],
                                    y=[cat] * len(cat_df),
                                    mode='markers',
                                    name=cat,
                                    marker=dict(size=10)
                                ),
                                row=2, col=2
                            )
                
                fig.update_layout(height=600, showlegend=True)
                st.plotly_chart(fig, use_container_width=True)
        else:
            st.info("No queries yet. Start chatting to see analytics!")
    
    with tabs[2]:
        # Knowledge Base Explorer
        st.markdown("### 📚 Knowledge Base Explorer")
        
        if st.session_state.agent:
            # Search in KB
            kb_search = st.text_input("Search knowledge base:", placeholder="Enter keywords...")
            
            if kb_search:
                with st.spinner("Searching..."):
                    results = st.session_state.agent.retriever.hybrid_search(kb_search, k=5)
                    
                    if results:
                        for doc in results:
                            with st.expander(f"📄 {doc.title}"):
                                st.markdown(f"**Category:** {doc.category}")
                                st.markdown(f"**Description:** {doc.content}")
                                st.markdown("**Solutions:**")
                                for i, solution in enumerate(doc.solutions, 1):
                                    st.write(f"{i}. {solution}")
                                st.markdown(f"**Keywords:** {', '.join(doc.keywords)}")
                    else:
                        st.warning("No results found")
            
            # Browse by category
            st.markdown("#### Browse by Category")
            
            categories = {}
            for doc in st.session_state.agent.retriever.documents:
                if doc.category not in categories:
                    categories[doc.category] = []
                categories[doc.category].append(doc)
            
            selected_category = st.selectbox("Select category:", list(categories.keys()))
            
            if selected_category:
                st.markdown(f"#### {selected_category} ({len(categories[selected_category])} documents)")
                
                for doc in categories[selected_category]:
                    with st.expander(f"📄 {doc.title}"):
                        st.markdown(f"**Description:** {doc.content}")
                        st.markdown("**Solutions:**")
                        for i, solution in enumerate(doc.solutions, 1):
                            st.write(f"{i}. {solution}")

# Footer
st.divider()
st.markdown("""
    <div style='text-align: center; color: #666; padding: 1rem;'>
    <b>Tech Support AI v2.0</b> | Powered by Groq LPU™ | Hybrid RAG with BM25 + Dense Retrieval
    <br>Knowledge Base: Microsoft Docs, Stack Overflow, IT Best Practices
    </div>
    """, unsafe_allow_html=True)