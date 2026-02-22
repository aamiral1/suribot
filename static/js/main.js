
const messageSendButton = document.querySelector(".vb .vb-footer button.vb-send");
const inputBar = document.querySelector(".vb .vb-footer input.vb-input");
const messageWindow = document.querySelector(".vb-messages")

const BASE = "http://127.0.0.1:5000/"

async function addMessage(message, role){
    const row = document.createElement("div");
    row.classList.add("vb-msg-row");

    if (role == "agent"){
        // add avatar
        const avatar = document.createElement("div");
        avatar.classList.add("vb-msg-avatar");
        avatar.setAttribute('aria-hidden', true);
        avatar.textContent = "✦";
        row.appendChild(avatar);
    
        
    }
    else if (role == "user"){
        row.classList.add("user");
    }

    // chat bubble
    const bubble = document.createElement("div");
    bubble.classList.add("vb-bubble", role);
    bubble.textContent = message;

    row.appendChild(bubble);

    messageWindow.append(row);

    // scroll to top
    messageWindow.scrollTop = messageWindow.scrollHeight;
}

async function sendMessage(){
    const input = inputBar.value.trim();

    if(!input){
        return;
    }
    // console.log(input);

    addMessage(input, "user");
    inputBar.value = "";

    // TESTING - REMOVE LATER
    // addMessage("How can I help you?", "agent");

    // POST REQUEST TESTING
    try {
        const res = await fetch(BASE + '/api', {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify({message: input})
        });
        const data = await res.json();

        if (!res.ok) {
            console.log(data.description);
            return;
        }
        const gpt_response = data['response'];

        console.log("GPT: " + gpt_response);

        addMessage(gpt_response, "agent");

    } catch (error) {
        console.log(error);
    }
}


messageSendButton.addEventListener("click", sendMessage);