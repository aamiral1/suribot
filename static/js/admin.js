// ── Elements ─────────────────────────────────────────────────
const form           = document.getElementById('upload_form');
const fileInput      = document.getElementById('file-input');
const uploadBtn      = document.getElementById('upload-btn');
const dropZone       = document.getElementById('drop-zone');
const uploadQueue    = document.getElementById('upload-queue');
const extractBtn     = document.getElementById('extract_btn');
const addToKbBtn     = document.getElementById('add_to_kb_btn');
const statusField    = document.getElementById('status_field');
const textField      = document.getElementById('text_field');
const refreshDocsBtn  = document.getElementById('refresh_docs_btn');
const docsTbody       = document.getElementById('docs-tbody');
const docCountTag     = document.getElementById('doc-count-tag');
const statDocCount    = document.getElementById('stat-doc-count');
const statStorageUsed = document.getElementById('stat-storage-used');

let latestDocId = null;
let interval    = null;

// tracks docId → queue item element for badge updates
const queueItems = new Map();


// ── File input triggers ───────────────────────────────────────
uploadBtn.addEventListener('click', () => fileInput.click());
dropZone.addEventListener('click',  () => fileInput.click());

fileInput.addEventListener('change', () => {
  if (fileInput.files.length > 0) submitForm();
});


// ── Drag and drop ─────────────────────────────────────────────
dropZone.addEventListener('dragover', (e) => {
  e.preventDefault();
  dropZone.classList.add('drag-over');
});

dropZone.addEventListener('dragleave', () => {
  dropZone.classList.remove('drag-over');
});

dropZone.addEventListener('drop', (e) => {
  e.preventDefault();
  dropZone.classList.remove('drag-over');

  const files = e.dataTransfer.files;
  if (files.length > 0) {
    const dt = new DataTransfer();
    dt.items.add(files[0]);
    fileInput.files = dt.files;
    submitForm();
  }
});


// ── Form submission ───────────────────────────────────────────
form.addEventListener('submit', async (e) => {
  e.preventDefault();
  await submitForm();
});

async function submitForm() {
  const fd = new FormData(form);

  try {
    const res  = await fetch('/admin', { method: 'POST', body: fd });
    const data = await res.json();

    if (!res.ok || data.status !== 'ok') {
      console.log('Upload failed:', data.error);
      return;
    }

    latestDocId = data.doc_id;

    const file     = fileInput.files[0];
    const fileName = file?.name  || 'Unknown file';
    const fileSize = formatSize(file?.size);

    addQueueItem(latestDocId, fileName, fileSize);
    console.log('Uploaded:', data);

  } catch (err) {
    console.error(err);
  }
}


// ── Queue item management ─────────────────────────────────────
function fileIcon(name) {
  const ext = name.split('.').pop().toLowerCase();
  if (ext === 'pdf')                        return { emoji: '📄', cls: 'qi-icon--pdf' };
  if (['doc', 'docx'].includes(ext))        return { emoji: '📝', cls: 'qi-icon--doc' };
  if (['png', 'jpg', 'jpeg'].includes(ext)) return { emoji: '🖼️', cls: 'qi-icon--img' };
  return { emoji: '📃', cls: 'qi-icon--txt' };
}

function addQueueItem(docId, fileName, fileSize) {
  const { emoji, cls } = fileIcon(fileName);

  const item = document.createElement('div');
  item.className     = 'queue-item';
  item.dataset.docId = docId;
  item.innerHTML = `
    <div class="qi-icon ${cls}">${emoji}</div>
    <div class="qi-info">
      <div class="qi-name">${fileName}</div>
      <div class="qi-meta">${fileSize} · Uploaded just now</div>
    </div>
    <span class="qi-badge qi-badge--created">○ Created</span>
    <button class="qi-remove" title="Remove">✕</button>
  `;

  item.querySelector('.qi-remove').addEventListener('click', () => item.remove());

  uploadQueue.appendChild(item);
  queueItems.set(docId, item);
}

function updateQueueBadge(docId, status) {
  const item = queueItems.get(docId);
  if (!item) return;

  const badge = item.querySelector('.qi-badge');
  const map = {
    created:    { cls: 'qi-badge--created',    text: '○ Created' },
    processing: { cls: 'qi-badge--processing', text: '⟳ Processing' },
    success:    { cls: 'qi-badge--success',    text: '✓ Success' },
    failed:     { cls: 'qi-badge--failed',     text: '✕ Failed' },
  };

  const entry = map[status];
  if (!entry) return;

  badge.className   = `qi-badge ${entry.cls}`;
  badge.textContent = entry.text;
}

function formatSize(bytes) {
  if (!bytes) return '';
  if (bytes < 1024)           return `${bytes} B`;
  if (bytes < 1024 * 1024)    return `${(bytes / 1024).toFixed(1)} KB`;
  return `${(bytes / (1024 * 1024)).toFixed(1)} MB`;
}


// ── API helpers ───────────────────────────────────────────────
async function fetchDocStatus(docId) {
  const res  = await fetch(`/document/${docId}/status`);
  const data = await res.json();

  if (!res.ok) {
    console.log('Problem fetching document status');
    return null;
  }

  return data;
}

async function fetchExtractedText(docId) {
  const res  = await fetch(`/document/${docId}/text`);
  const data = await res.json();

  if (!res.ok) {
    console.log('Problem fetching extracted text');
    return null;
  }

  return data;
}


// ── Polling ───────────────────────────────────────────────────
function startPolling(docId) {
  if (interval) clearInterval(interval);

  interval = setInterval(async () => {
    const data = await fetchDocStatus(docId);

    if (!data) {
      statusField.textContent = 'error';
      updateQueueBadge(docId, 'failed');
      clearInterval(interval);
      return;
    }

    const status = data.status;
    statusField.textContent = status;
    updateQueueBadge(docId, status);

    if (status === 'failed' || status === 'error') {
      clearInterval(interval);
      return;
    }

    if (status === 'processing' || status === 'created') {
      return;
    }

    if (status === 'success' && data.has_text === true) {
      clearInterval(interval);

      const textData = await fetchExtractedText(docId);
      if (!textData) {
        statusField.textContent = 'error';
        return;
      }

      textField.textContent = textData.text || '';
      addToKbBtn.style.display = 'inline-flex';
    }
  }, 2000);
}


// ── Documents table ───────────────────────────────────────────
const typeIconMap = {
  PDF:  { emoji: '📄', bg: 'rgba(239,68,68,0.1)' },
  DOCX: { emoji: '📝', bg: 'rgba(59,130,246,0.1)' },
  TXT:  { emoji: '📃', bg: 'rgba(107,114,128,0.1)' },
  MD:   { emoji: '📃', bg: 'rgba(107,114,128,0.1)' },
  PNG:  { emoji: '🖼️', bg: 'rgba(16,185,129,0.1)' },
  JPG:  { emoji: '🖼️', bg: 'rgba(16,185,129,0.1)' },
};

// parses formatted size strings ("1.20 MB", "340.00 KB") back to bytes
function parseSize(sizeStr) {
  if (!sizeStr) return 0;
  const units = { B: 1, KB: 1024, MB: 1024 ** 2, GB: 1024 ** 3, TB: 1024 ** 4 };
  const match = sizeStr.trim().match(/^([\d.]+)\s*(B|KB|MB|GB|TB)$/i);
  if (!match) return 0;
  return parseFloat(match[1]) * (units[match[2].toUpperCase()] || 1);
}

async function loadDocumentsTable() {
  try {
    const res  = await fetch('/documents');
    const data = await res.json();

    if (!res.ok) {
      console.log('Failed to load documents:', data.error);
      return;
    }

    const successful = data.documents.filter(doc => doc.in_kb === true);

    // deduplicate by file_name, keeping the most recent entry per name
    const seen = new Map();
    successful
      .sort((a, b) => new Date(a.uploaded_date) - new Date(b.uploaded_date))
      .forEach(doc => seen.set(doc.file_name, doc));
    const docs = Array.from(seen.values());

    docsTbody.innerHTML = '';

    docs.forEach((doc) => {
      const { emoji, bg } = typeIconMap[doc.file_type] || { emoji: '📄', bg: 'rgba(107,114,128,0.1)' };
      const date = doc.uploaded_date ? doc.uploaded_date.slice(0, 10) : '—';

      const tr = document.createElement('tr');
      tr.innerHTML = `
        <td><div class="td-file">
          <div class="td-file-icon" style="background:${bg}">${emoji}</div>
          <span class="td-file-name">${doc.file_name || '—'}</span>
        </div></td>
        <td class="td-muted">${doc.file_type || '—'}</td>
        <td class="td-muted">${doc.file_size || '—'}</td>
        <td class="td-muted">${date}</td>
        <td class="td-muted">${doc.status || '—'}</td>
      `;
      docsTbody.appendChild(tr);
    });

    const totalBytes = docs.reduce((sum, doc) => sum + parseSize(doc.file_size), 0);

    docCountTag.textContent     = `${docs.length} Active`;
    statDocCount.textContent    = docs.length;
    statStorageUsed.innerHTML   = formatSize(totalBytes).replace(/\s(\S+)$/, '<span style="font-size:16px;font-weight:500">$1</span>');

  } catch (err) {
    console.error('Error loading documents table:', err);
  }
}

refreshDocsBtn.addEventListener('click', () => {
  textField.textContent       = '';
  statusField.textContent     = '';
  addToKbBtn.style.display    = 'none';
  addToKbBtn.disabled         = false;
  addToKbBtn.textContent      = '＋ Add to Knowledge Base';
  loadDocumentsTable();
});

// load on page open
loadDocumentsTable();


// ── Add to Knowledge Base button ──────────────────────────────
addToKbBtn.addEventListener('click', async () => {
  try {
    const res  = await fetch(`/document/${latestDocId}/add-to-kb`, { method: 'POST' });
    const data = await res.json();

    if (!res.ok) {
      console.log('Failed to add to KB:', data.message);
      return;
    }

    // TODO: chunk → vectorise → update vector DB

    addToKbBtn.disabled     = true;
    addToKbBtn.textContent  = '✓ Added to Knowledge Base';
    loadDocumentsTable();

  } catch (err) {
    console.error(err);
  }
});


// ── Extract button ────────────────────────────────────────────
extractBtn.addEventListener('click', async () => {
  try {
    console.log('Extracting doc:', latestDocId);

    const res = await fetch('/extract-text', {
      method: 'POST',
      headers: { 'Content-Type': 'application/json' },
      body: JSON.stringify({ doc_id: latestDocId }),
    });

    const data = await res.json();

    if (!res.ok) {
      console.log(data.message || data.error || 'Failed to start extraction');
      statusField.textContent = 'error';
      return;
    }

    if (
      data.status === 'began processing'  ||
      data.status === 'already processing' ||
      data.status === 'already extracted'
    ) {
      statusField.textContent = data.status;
      startPolling(latestDocId);
      return;
    }

    console.log('Unexpected response:', data);
    statusField.textContent = 'error';

  } catch (err) {
    console.error(err);
    statusField.textContent = 'error';
  }
});
