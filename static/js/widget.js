(function () {
  var CHAT_URL = "http://localhost:5000/embed";

  var WIDGET_ID = "suribot-widget";

  if (document.getElementById(WIDGET_ID)) return;

  var style = document.createElement("style");
  style.textContent = [
    "#" + WIDGET_ID + "-btn {",
    "  position: fixed; bottom: 24px; right: 24px; z-index: 2147483647;",
    "  width: 56px; height: 56px; border-radius: 50%;",
    "  background: #6c2fff; color: #fff; font-size: 22px;",
    "  border: none; cursor: pointer; box-shadow: 0 4px 16px rgba(0,0,0,0.25);",
    "  display: flex; align-items: center; justify-content: center;",
    "  transition: transform 0.2s;",
    "}",
    "#" + WIDGET_ID + "-btn:hover { transform: scale(1.08); }",
    "#" + WIDGET_ID + "-panel {",
    "  position: fixed; bottom: 92px; right: 24px; z-index: 2147483646;",
    "  width: 360px; height: 620px; border-radius: 16px;",
    "  box-shadow: 0 8px 32px rgba(0,0,0,0.22); border: none;",
    "  transition: opacity 0.2s, transform 0.2s; transform-origin: bottom right;",
    "}",
    "#" + WIDGET_ID + "-panel.hidden {",
    "  opacity: 0; transform: scale(0.9); pointer-events: none;",
    "}",
    "@media (max-width: 480px) {",
    "  #" + WIDGET_ID + "-panel {",
    "    width: 100vw; height: 100vh; bottom: 0; right: 0; border-radius: 0;",
    "  }",
    "}",
  ].join("\n");
  document.head.appendChild(style);

  var iframe = document.createElement("iframe");
  iframe.id = WIDGET_ID + "-panel";
  iframe.src = CHAT_URL;
  iframe.classList.add("hidden");
  iframe.setAttribute("title", "Suri Marketing Chat");
  iframe.setAttribute("allow", "clipboard-write");
  document.body.appendChild(iframe);

  var btn = document.createElement("button");
  btn.id = WIDGET_ID + "-btn";
  btn.setAttribute("aria-label", "Open chat");
  btn.innerHTML = "&#10022;";
  document.body.appendChild(btn);

  var open = false;
  btn.addEventListener("click", function () {
    open = !open;
    if (open) {
      iframe.classList.remove("hidden");
      btn.setAttribute("aria-label", "Close chat");
    } else {
      iframe.classList.add("hidden");
      btn.setAttribute("aria-label", "Open chat");
    }
  });
})();
