
function generateUUID() {
    if (crypto.randomUUID) return crypto.randomUUID();
    return 'xxxxxxxx-xxxx-4xxx-yxxx-xxxxxxxxxxxx'.replace(/[xy]/g, function(c) {
        const r = Math.random() * 16 | 0;
        return (c === 'x' ? r : (r & 0x3 | 0x8)).toString(16);
    });
}
if (!sessionStorage.getItem('session_id')) {
    sessionStorage.setItem('session_id', generateUUID());
}
const sessionId = sessionStorage.getItem('session_id');

const messageSendButton = document.querySelector(".vb .vb-footer button.vb-send");
const inputBar = document.querySelector(".vb .vb-footer input.vb-input");
const messageWindow = document.querySelector(".vb-messages")

async function addMessage(message, role){
    const row = document.createElement("div");
    row.classList.add("vb-msg-row");

    if (role == "agent"){
        const avatar = document.createElement("div");
        avatar.classList.add("vb-msg-avatar");
        avatar.setAttribute('aria-hidden', true);
        avatar.textContent = "✦";
        row.appendChild(avatar);
    
        
    }
    else if (role == "user"){
        row.classList.add("user");
    }

    const bubble = document.createElement("div");
    bubble.classList.add("vb-bubble", role);
    if (role === 'agent' && typeof marked !== 'undefined') {
        bubble.innerHTML = marked.parse(message);
    } else {
        bubble.textContent = message;
    }

    row.appendChild(bubble);

    messageWindow.append(row);
    messageWindow.scrollTop = messageWindow.scrollHeight;
}

function showTypingIndicator() {
    const row = document.createElement("div");
    row.classList.add("vb-msg-row", "vb-typing-row");

    const avatar = document.createElement("div");
    avatar.classList.add("vb-msg-avatar");
    avatar.setAttribute("aria-hidden", true);
    avatar.textContent = "✦";

    const bubble = document.createElement("div");
    bubble.classList.add("vb-bubble", "agent");

    const loader = document.createElement("div");
    loader.classList.add("loader");
    bubble.appendChild(loader);

    row.appendChild(avatar);
    row.appendChild(bubble);
    messageWindow.appendChild(row);
    messageWindow.scrollTop = messageWindow.scrollHeight;
    return row;
}

async function sendMessage(){
    const input = inputBar.value.trim();

    if(!input){
        return;
    }

    addMessage(input, "user");
    inputBar.value = "";
    messageSendButton.disabled = true;

    const typingRow = showTypingIndicator();

    try {
        const res = await fetch('/api', {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({message: input, session_id: sessionId})
        });
        const data = await res.json();

        typingRow.remove();

        if (!res.ok) {
            return;
        }
        const gpt_response = data['response'];

        addMessage(gpt_response, "agent");

    } catch (error) {
        typingRow.remove();
    } finally {
        messageSendButton.disabled = false;
    }
}


messageSendButton.addEventListener("click", sendMessage);

const ctaButton = document.querySelector(".vb-cta");
ctaButton.addEventListener("click", () => {
    inputBar.value = "I want to book a free discovery call";
    sendMessage();
});