/* ── Password toggle ──────────────────────────────────────────── */
    const pwInput  = document.getElementById('password');
    const pwToggle = document.getElementById('pw-toggle');
    const eyeIcon  = document.getElementById('eye-icon');
 
    const eyeOpen   = `<path d="M2 12s3-7 10-7 10 7 10 7-3 7-10 7-10-7-10-7Z"/><circle cx="12" cy="12" r="3"/>`;
    const eyeClosed = `<path d="M17.94 17.94A10.07 10.07 0 0 1 12 20c-7 0-11-8-11-8a18.45 18.45 0 0 1 5.06-5.94"/><path d="M9.9 4.24A9.12 9.12 0 0 1 12 4c7 0 11 8 11 8a18.5 18.5 0 0 1-2.16 3.19"/><line x1="1" y1="1" x2="23" y2="23"/>`;
 
    pwToggle.addEventListener('click', () => {
      const showing = pwInput.type === 'text';
      pwInput.type = showing ? 'password' : 'text';
      eyeIcon.innerHTML = showing ? eyeOpen : eyeClosed;
      pwToggle.setAttribute('aria-label', showing ? 'Show password' : 'Hide password');
    });
 
    /* ── Form submit ──────────────────────────────────────────────── */
    const form       = document.getElementById('login-form');
    const submitBtn  = document.getElementById('submit-btn');
    const errorBanner = document.getElementById('error-banner');
    const errorText   = document.getElementById('error-text');
 
    function showError(msg) {
      errorText.textContent = msg;
      errorBanner.classList.remove('visible');
      // Force reflow so animation re-triggers
      void errorBanner.offsetWidth;
      errorBanner.classList.add('visible');
    }
 
    function hideError() {
      errorBanner.classList.remove('visible');
    }
 
    form.addEventListener('submit', async (e) => {
      e.preventDefault();
      hideError();
 
      const username = document.getElementById('username').value.trim();
      const password = document.getElementById('password').value;

      if (!username) { showError('Please enter your username.'); return; }
      if (!password) { showError('Please enter your password.'); return; }

      submitBtn.classList.add('loading');

      try {
        const res = await fetch('/login', {
          method: 'POST',
          headers: { 'Content-Type': 'application/json' },
          body: JSON.stringify({ username, password }),
        });
        if (res.ok) {
          window.location.href = '/analytics';
        } else {
          const data = await res.json();
          showError(data.message || 'Invalid username or password.');
        }
      } catch {
        showError('Network error. Please try again.');
      } finally {
        submitBtn.classList.remove('loading');
      }
    });

    // Hide error when user starts typing
    document.getElementById('username').addEventListener('input', hideError);
    document.getElementById('password').addEventListener('input', hideError);