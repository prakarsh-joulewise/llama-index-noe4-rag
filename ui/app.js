// Global Configuration & State
const loc = window.location;
let host = loc.host || 'localhost:8000';

// If the UI is being served from port 3006, redirect API/WS requests to the backend at port 5005
if (loc.port === '3006') {
    host = `${loc.hostname}:5005`;
}

const protocol = loc.protocol === 'https:' ? 'https:' : 'http:';
const wsProtocol = loc.protocol === 'https:' ? 'wss:' : 'ws:';

const API_STATUS_URL = `${protocol}//${host}/api/status`;
const WS_CHAT_URL = `${wsProtocol}//${host}/api/chat`;

let chatHistory = [];
let ws = null;
let currentAssistantBubble = null;
let currentStatusLog = null;
let accumulatedResponseText = '';

// DOM Elements
const chatMessages = document.getElementById('chat-messages');
const chatForm = document.getElementById('chat-form');
const userInput = document.getElementById('user-input');
const btnSend = document.getElementById('btn-send');
const btnClearChat = document.getElementById('btn-clear-chat');
const welcomeMessage = document.getElementById('welcome-message');
const sourcesContainer = document.getElementById('sources-container');
const sourcesList = document.getElementById('sources-list');
const sourceCount = document.getElementById('source-count');
const btnToggleSources = document.getElementById('btn-toggle-sources');

// Status indicators
const indicatorApi = document.querySelector('#status-api .status-indicator');
const labelApi = document.querySelector('#status-api .status-label');
const indicatorNeo4j = document.querySelector('#status-neo4j .status-indicator');
const labelNeo4j = document.querySelector('#status-neo4j .status-label');
const activeModelLabel = document.getElementById('active-model');

// Set Markdown configuration (disable mangling/header IDs to prevent errors in newer versions)
if (window.marked) {
    window.marked.setOptions({
        breaks: true,
        gfm: true
    });
}

// -------------------------------------------------------------
// Initialize App
// -------------------------------------------------------------
document.addEventListener('DOMContentLoaded', () => {
    loadChatHistory();
    checkSystemStatus();
    connectWebSocket();
    
    // Poll system status every 5 seconds
    setInterval(checkSystemStatus, 5000);
    
    // Auto-grow input textarea
    userInput.addEventListener('input', autoGrowInput);
    
    // Handle Enter to send
    userInput.addEventListener('keydown', (e) => {
        if (e.key === 'Enter' && !e.shiftKey) {
            e.preventDefault();
            chatForm.dispatchEvent(new Event('submit'));
        }
    });

    // Handle suggestion clicks
    document.addEventListener('click', (e) => {
        const card = e.target.closest('.suggestion-card, .suggest-btn');
        if (card) {
            const query = card.getAttribute('data-query');
            if (query) {
                sendQuery(query);
            }
        }
    });

    // Toggle sources drawer
    btnToggleSources.addEventListener('click', () => {
        const isCollapsed = sourcesList.style.display === 'none';
        sourcesList.style.display = isCollapsed ? 'grid' : 'none';
        btnToggleSources.querySelector('i').className = isCollapsed 
            ? 'fa-solid fa-chevron-down' 
            : 'fa-solid fa-chevron-up';
    });

    // Clear conversation
    btnClearChat.addEventListener('click', clearConversation);
});

// -------------------------------------------------------------
// System Health Monitoring
// -------------------------------------------------------------
async function checkSystemStatus() {
    try {
        const res = await fetch(API_STATUS_URL);
        if (!res.ok) throw new Error('API return non-200');
        const data = await res.json();
        
        // Update API Label
        indicatorApi.className = 'status-indicator online';
        labelApi.textContent = 'Backend API: Online';
        
        // Update Neo4j Label
        if (data.neo4j_connected) {
            indicatorNeo4j.className = 'status-indicator online';
            labelNeo4j.textContent = `Neo4j DB: Connected (${data.neo4j_chunks} chunks)`;
        } else {
            indicatorNeo4j.className = 'status-indicator offline';
            labelNeo4j.textContent = 'Neo4j DB: Offline';
        }
        
        // Update Model Badge
        activeModelLabel.textContent = data.llm_model;
        
    } catch (err) {
        indicatorApi.className = 'status-indicator offline';
        labelApi.textContent = 'Backend API: Offline';
        indicatorNeo4j.className = 'status-indicator offline';
        labelNeo4j.textContent = 'Neo4j DB: Offline';
    }
}

// -------------------------------------------------------------
// WebSocket Connection Manager
// -------------------------------------------------------------
function connectWebSocket() {
    if (ws && ws.readyState === WebSocket.OPEN) return;

    console.log(`[WebSocket] Connecting to: ${WS_CHAT_URL}`);
    ws = new WebSocket(WS_CHAT_URL);

    ws.onopen = () => {
        console.log('[WebSocket] Connection established.');
    };

    ws.onmessage = (event) => {
        const data = JSON.parse(event.data);
        handleServerMessage(data);
    };

    ws.onerror = (error) => {
        console.error('[WebSocket] Error:', error);
    };

    ws.onclose = (event) => {
        console.log('[WebSocket] Connection closed. Reconnecting in 3s...', event.reason);
        setTimeout(connectWebSocket, 3000);
    };
}

// -------------------------------------------------------------
// Chat Operations
// -------------------------------------------------------------
chatForm.addEventListener('submit', (e) => {
    e.preventDefault();
    const message = userInput.value.trim();
    if (!message) return;
    
    sendQuery(message);
    userInput.value = '';
    userInput.style.height = 'auto'; // Reset height
});

function sendQuery(message) {
    if (ws.readyState !== WebSocket.OPEN) {
        appendMessage('system', 'Unable to send query. Reconnecting to server, please wait...');
        connectWebSocket();
        return;
    }

    // Hide welcome card if visible
    if (welcomeMessage) {
        welcomeMessage.style.display = 'none';
    }

    // Clear sources drawer from previous turn
    sourcesContainer.style.display = 'none';
    sourcesList.innerHTML = '';

    // Append user message
    appendMessage('user', message);

    // Push user turn to history
    chatHistory.push({ role: 'user', content: message });
    saveChatHistory();

    // Append assistant typing skeleton
    createAssistantBubble();

    // Send payload over WebSocket
    const payload = {
        message: message,
        history: chatHistory
    };
    ws.send(JSON.stringify(payload));
    disableInput(true);
}

function handleServerMessage(data) {
    switch (data.type) {
        case 'status':
            // Update typing indicator text log
            if (currentStatusLog) {
                currentStatusLog.innerHTML = `<i class="fa-solid fa-spinner"></i> ${data.content}`;
            }
            break;
            
        case 'sources':
            // Render retrieved sources
            renderSources(data.content);
            break;
            
        case 'token':
            // Remove status/typing indicators on first token
            if (currentStatusLog) {
                currentStatusLog.remove();
                currentStatusLog = null;
            }
            const typingInd = currentAssistantBubble.querySelector('.typing-indicator');
            if (typingInd) {
                typingInd.remove();
            }

            // Stream token
            accumulatedResponseText += data.content;
            
            // Render markdown live
            if (window.marked) {
                currentAssistantBubble.innerHTML = window.marked.parse(accumulatedResponseText);
            } else {
                currentAssistantBubble.textContent = accumulatedResponseText;
            }
            scrollToBottom();
            break;
            
        case 'done':
            // Save complete assistant turn to history
            chatHistory.push({ role: 'assistant', content: accumulatedResponseText });
            saveChatHistory();
            
            // Clean up state
            resetTurnState();
            disableInput(false);
            break;
            
        case 'error':
            if (currentStatusLog) currentStatusLog.remove();
            const ind = currentAssistantBubble.querySelector('.typing-indicator');
            if (ind) ind.remove();
            
            currentAssistantBubble.innerHTML = `<div style="color: var(--danger);"><i class="fa-solid fa-triangle-exclamation"></i> ${data.content}</div>`;
            resetTurnState();
            disableInput(false);
            break;
    }
}

function resetTurnState() {
    currentAssistantBubble = null;
    currentStatusLog = null;
    accumulatedResponseText = '';
}

function disableInput(disabled) {
    userInput.disabled = disabled;
    btnSend.disabled = disabled;
    if (!disabled) {
        userInput.focus();
    }
}

// -------------------------------------------------------------
// Message Rendering
// -------------------------------------------------------------
function appendMessage(role, text) {
    const msgElement = document.createElement('div');
    msgElement.className = `message ${role}`;

    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.innerHTML = role === 'user' ? '<i class="fa-solid fa-user"></i>' : '<i class="fa-solid fa-robot"></i>';

    const bubble = document.createElement('div');
    bubble.className = 'bubble';
    
    if (role === 'user') {
        bubble.textContent = text;
    } else {
        bubble.innerHTML = window.marked ? window.marked.parse(text) : text;
    }

    msgElement.appendChild(avatar);
    msgElement.appendChild(bubble);
    chatMessages.appendChild(msgElement);
    scrollToBottom();
}

function createAssistantBubble() {
    const msgElement = document.createElement('div');
    msgElement.className = 'message assistant';

    const avatar = document.createElement('div');
    avatar.className = 'avatar';
    avatar.innerHTML = '<i class="fa-solid fa-robot"></i>';

    const bubble = document.createElement('div');
    bubble.className = 'bubble';

    // Status Log for search activity
    currentStatusLog = document.createElement('div');
    currentStatusLog.className = 'status-log';
    currentStatusLog.innerHTML = '<i class="fa-solid fa-spinner"></i> Initiating hybrid search...';
    
    // Typing indicator
    const typingIndicator = document.createElement('div');
    typingIndicator.className = 'typing-indicator';
    typingIndicator.innerHTML = '<span></span><span></span><span></span>';

    bubble.appendChild(currentStatusLog);
    bubble.appendChild(typingIndicator);
    
    msgElement.appendChild(avatar);
    msgElement.appendChild(bubble);
    chatMessages.appendChild(msgElement);
    
    currentAssistantBubble = bubble;
    scrollToBottom();
}

function renderSources(sources) {
    if (!sources || sources.length === 0) return;
    
    sourceCount.textContent = sources.length;
    sourcesContainer.style.display = 'block';
    sourcesList.innerHTML = '';
    sourcesList.style.display = 'grid'; // ensure expanded
    btnToggleSources.querySelector('i').className = 'fa-solid fa-chevron-down';

    sources.forEach(src => {
        const item = document.createElement('div');
        item.className = 'source-item';
        
        // Clean up score format
        const scorePct = (src.score * 100).toFixed(0);
        
        item.innerHTML = `
            <div class="source-doc" title="${src.document_name}">
                <i class="fa-solid fa-file-pdf" style="color: var(--danger);"></i> ${src.document_name}
            </div>
            <div class="source-meta">
                <span>State: <strong>${src.state}</strong> | FY: <strong>${src.year}</strong></span>
                <span style="color: var(--accent);">Match: ${scorePct}%</span>
            </div>
            <div class="source-path" title="${src.header_path}">
                Path: ${src.header_path}
            </div>
        `;
        sourcesList.appendChild(item);
    });
}

// -------------------------------------------------------------
// Local Storage Persistence
// -------------------------------------------------------------
function saveChatHistory() {
    localStorage.setItem('joulewise_history', JSON.stringify(chatHistory));
}

function loadChatHistory() {
    const saved = localStorage.getItem('joulewise_history');
    if (saved) {
        chatHistory = JSON.parse(saved);
        if (chatHistory.length > 0) {
            if (welcomeMessage) welcomeMessage.style.display = 'none';
            chatHistory.forEach(turn => {
                appendMessage(turn.role, turn.content);
            });
        }
    }
}

function clearConversation() {
    chatHistory = [];
    localStorage.removeItem('joulewise_history');
    
    // Clear DOM
    const welcome = welcomeMessage.cloneNode(true);
    welcome.style.display = 'block';
    
    chatMessages.innerHTML = '';
    chatMessages.appendChild(welcome);
    
    sourcesContainer.style.display = 'none';
    sourcesList.innerHTML = '';
    
    resetTurnState();
    disableInput(false);
}

// -------------------------------------------------------------
// Helper Utilities
// -------------------------------------------------------------
function scrollToBottom() {
    chatMessages.scrollTop = chatMessages.scrollHeight;
}

function autoGrowInput() {
    userInput.style.height = 'auto';
    userInput.style.height = userInput.scrollHeight + 'px';
}
