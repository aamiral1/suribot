const extractBtnEl = document.getElementById('extract_btn');
const form = document.getElementById('upload_form');
const textFieldEl = document.getElementById('text_field');

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
        
        latest_doc_id = data.doc_id;
        console.log(latest_doc_id);

    } catch (error) {
        console.log(error)
    }
     
});

// prerequisite: doc-id to extract
function sendExtractDocRequest(doc_id){
    console.log("processing doc " + doc_id);

    // send post request to backend with doc_id

    return null;
}
    
extractBtnEl.addEventListener('click', async () => {
   try {
        const msg = {
            'doc_id': latest_doc_id // get doc_id
        };

        const res = await fetch('/extract-text', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json'
            },
            body: JSON.stringify(msg)
        });

        const data = await res.json();

        if(!res.ok || data.status !== 'ok'){
            console.log(data.error);
            return;
        }

        console.log(data);

        // update div with extractede text
        textFieldEl.innerHTML = data.extracted_text;
        console.log("Extracted text outputted successfully");
        
   } catch (error) {
        console.log(error)
   }
});