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

// Mobile UI Elements
const sidebar = document.querySelector('.sidebar');
const sidebarOverlay = document.getElementById('sidebar-overlay');
const btnMenuToggle = document.getElementById('btn-menu-toggle');
const btnSidebarClose = document.getElementById('btn-sidebar-close');

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

    // Mobile Sidebar helper functions
    const closeSidebar = () => {
        if (sidebar && sidebarOverlay) {
            sidebar.classList.remove('active');
            sidebarOverlay.classList.remove('active');
        }
    };

    // Mobile Sidebar Event Listeners
    if (btnMenuToggle && sidebar && sidebarOverlay) {
        btnMenuToggle.addEventListener('click', () => {
            sidebar.classList.add('active');
            sidebarOverlay.classList.add('active');
        });
    }

    if (btnSidebarClose) {
        btnSidebarClose.addEventListener('click', closeSidebar);
    }

    if (sidebarOverlay) {
        sidebarOverlay.addEventListener('click', closeSidebar);
    }

    // Handle suggestion clicks
    document.addEventListener('click', (e) => {
        const card = e.target.closest('.suggestion-card, .suggest-btn');
        if (card) {
            const query = card.getAttribute('data-query');
            if (query) {
                sendQuery(query);
                // Close sidebar on mobile after choosing a suggestion
                if (window.innerWidth <= 768) {
                    closeSidebar();
                }
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

    // Initialize Liquid Background Animation
    initLiquidBg();
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
    try {
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
    } catch (error) {
        console.warn('[WebSocket] Connection failed to initialize (Backend might be offline):', error);
        // Retry connection in 5 seconds
        setTimeout(connectWebSocket, 5000);
    }
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

// -------------------------------------------------------------
// Liquid Background Canvas Animation
// -------------------------------------------------------------
class LiquidBlob {
    constructor(width, height, color) {
        this.x = Math.random() * width;
        this.y = Math.random() * height;
        // Slow shifting velocities
        this.vx = (Math.random() - 0.5) * 0.6;
        this.vy = (Math.random() - 0.5) * 0.6;
        this.radius = Math.random() * 150 + 150; // large soft blobs
        this.color = color;
    }

    update(width, height) {
        this.x += this.vx;
        this.y += this.vy;

        // Bounce boundaries and correct position to avoid getting stuck
        if (this.x < 0) {
            this.x = 0;
            this.vx = Math.abs(this.vx);
        } else if (this.x > width) {
            this.x = width;
            this.vx = -Math.abs(this.vx);
        }

        if (this.y < 0) {
            this.y = 0;
            this.vy = Math.abs(this.vy);
        } else if (this.y > height) {
            this.y = height;
            this.vy = -Math.abs(this.vy);
        }
    }

    draw(ctx) {
        ctx.beginPath();
        const grad = ctx.createRadialGradient(this.x, this.y, 0, this.x, this.y, this.radius);
        grad.addColorStop(0, this.color);
        grad.addColorStop(1, 'rgba(0, 0, 0, 0)');
        ctx.fillStyle = grad;
        ctx.arc(this.x, this.y, this.radius, 0, Math.PI * 2);
        ctx.fill();
    }
}

function initLiquidBg() {
    console.log('[LiquidBG] Initializing liquid background...');
    const canvas = document.getElementById('liquid-bg-canvas');
    if (!canvas) {
        console.warn('[LiquidBG] Canvas element not found!');
        return;
    }
    const ctx = canvas.getContext('2d');
    if (!ctx) {
        console.warn('[LiquidBG] Could not retrieve 2D context from canvas!');
        return;
    }

    // Run canvas at a lower internal resolution for rendering performance (stretched by CSS + blur filter)
    const scaleFactor = 0.5;
    
    // Safely get width and height with window fallbacks
    let offsetW = canvas.offsetWidth;
    let offsetH = canvas.offsetHeight;
    if (!offsetW || offsetW === 0) offsetW = window.innerWidth || 1024;
    if (!offsetH || offsetH === 0) offsetH = window.innerHeight || 768;
    
    let width = canvas.width = offsetW * scaleFactor;
    let height = canvas.height = offsetH * scaleFactor;
    console.log(`[LiquidBG] Canvas resolution set to ${width}x${height}`);

    // Vibrant, higher-opacity colors for beautiful blending under the CSS blur filter
    const colors = [
        'rgba(99, 102, 241, 0.75)',  // --primary (Indigo)
        'rgba(6, 182, 212, 0.7)',    // --accent (Cyan)
        'rgba(139, 92, 246, 0.65)',  // Deep Violet
        'rgba(30, 58, 138, 0.7)'     // Deep Blue
    ];

    const blobs = colors.map(c => new LiquidBlob(width, height, c));

    function resize() {
        if (!canvas) return;
        let w = canvas.offsetWidth;
        let h = canvas.offsetHeight;
        if (!w || w === 0) w = window.innerWidth || 1024;
        if (!h || h === 0) h = window.innerHeight || 768;
        width = canvas.width = w * scaleFactor;
        height = canvas.height = h * scaleFactor;
    }

    window.addEventListener('resize', resize);

    function animate() {
        if (!canvas) return;
        ctx.clearRect(0, 0, width, height);
        
        // Use default source-over compositing, which works reliably on transparent backgrounds
        ctx.globalCompositeOperation = 'source-over';

        blobs.forEach(blob => {
            blob.update(width, height);
            blob.draw(ctx);
        });

        requestAnimationFrame(animate);
    }

    animate();
    console.log('[LiquidBG] Animation loop running.');
}
