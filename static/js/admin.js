const extractBtnEl = document.getElementById('extract_btn');
const form = document.getElementById('upload_form');
const textFieldEl = document.getElementById('text_field');
const statusFieldEl = document.getElementById('status_field');

let latest_doc_id;

form.addEventListener("submit", async(e) => {
    e.preventDefault();

    const fd = new FormData(form);

    try {
        const res = await fetch('/admin', {
            method: 'POST',
            body: fd
        })
        const data = await res.json();

        // check if upload failed
        if(!res.ok || data.status !== 'ok'){
            console.log("Upload failed");
            console.log(data.error);
            return;
        }
        
        latest_doc_id = data.doc_id
        console.log(data);

    } catch (error) {
        console.log(error)
    }
     
});

let interval = null;

async function fetchDocStatus(docId) {
  const res = await fetch(`/document/${docId}/status`);
  const data = await res.json();

  if (!res.ok) {
    console.log("Problem with fetching document status");
    return null;
  }

  return data;
}

async function fetchExtractedText(docId) {
  const res = await fetch(`/document/${docId}/text`);
  const data = await res.json();

  if (!res.ok) {
    console.log("Problem with fetching extracted text");
    return null;
  }

  return data;
}

async function startPolling(docId) {
  if (interval) {
    clearInterval(interval);
  }

  interval = setInterval(async () => {
    const data = await fetchDocStatus(docId); // call document status API

    if (!data) {
      statusFieldEl.textContent = "error";
      clearInterval(interval);
      return;
    }

    // Set UI text field to received doc status
    const status = data.status;
    statusFieldEl.textContent = status;

    if (status === "failed" || status === "error") {
      console.log("Document extraction failed");
      clearInterval(interval);
      return;
    }

    if (status === "processing" || status === "created") {
      return;
    }

    if (status === "success" && data.has_text === true) {
      clearInterval(interval);

      // If extraction was successful, call Extracted Text API
      const textData = await fetchExtractedText(docId);

      if (!textData) {
        statusFieldEl.textContent = "error";
        return;
      }

      textFieldEl.textContent = textData.text || ""; // Set UI field to extracted text 
      return;
    }

    // unexpected case
    statusFieldEl.textContent = "error";
    clearInterval(interval);
  }, 2000);
}

extractBtnEl.addEventListener("click", async () => {
  try {
    console.log("Latest" + latest_doc_id);

    const msg = {"doc_id": latest_doc_id};

    const res = await fetch("/extract-text", {
      method: "POST",
      headers: {
        "Content-Type": "application/json",
      },
      body: JSON.stringify(msg),
    });

    const data = await res.json();

    if (!res.ok) {
      console.log("hit json res not ok")
      console.log(data.message || data.error || "Failed to start extraction");
      statusFieldEl.textContent = "error";
      return;
    }

    if (
      data.status === "began processing" ||
      data.status === "already processing" ||
      data.status === "already extracted"
    ) {
      statusFieldEl.textContent = data.status;
      startPolling(latest_doc_id);
      return;
    }

    console.log("Unexpected response:", data);
    statusFieldEl.textContent = "error";
  } catch (error) {
    console.log(error);
    statusFieldEl.textContent = "error";
  }
});