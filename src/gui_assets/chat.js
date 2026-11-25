const compose = document.querySelector("#compose");
const prompt = document.querySelector("#prompt");
const send = document.querySelector("#send");
const newChat = document.querySelector("#new-chat");
const themeToggle = document.querySelector("#theme-toggle");
const empty = document.querySelector("#empty");
const messages = document.querySelector("#messages");
const transcript = document.querySelector("#transcript");
const status = document.querySelector("#status");
const error = document.querySelector("#error");

const history = [];
const maxMessages = 21;
const themeKey = "reasoning-lab-theme";
let ready = false;
let generating = false;

function setTheme(theme) {
  document.documentElement.dataset.theme = theme;
  themeToggle.textContent = theme === "light" ? "Dark theme" : "Light theme";
  try {
    localStorage.setItem(themeKey, theme);
  } catch {
    // The current theme still works when storage is disabled.
  }
}

let storedTheme = null;
try {
  storedTheme = localStorage.getItem(themeKey);
} catch {
  // Default to the editor's dark theme.
}
setTheme(storedTheme === "light" ? "light" : "dark");
themeToggle.addEventListener("click", () => {
  setTheme(document.documentElement.dataset.theme === "dark" ? "light" : "dark");
});

function setStatus(label, state) {
  status.textContent = label;
  status.dataset.state = state;
}

function setError(message) {
  error.textContent = message;
  error.hidden = !message;
}

function updateControls() {
  prompt.disabled = !ready;
  prompt.readOnly = generating;
  send.disabled = !ready || generating || !prompt.value.trim();
  newChat.disabled = generating || history.length === 0;
}

function element(tag, className, text) {
  const node = document.createElement(tag);
  if (className) node.className = className;
  if (text !== undefined) node.textContent = text;
  return node;
}

function copyButton(text, label) {
  const button = element("button", "copy", label);
  button.type = "button";
  button.addEventListener("click", async () => {
    try {
      await navigator.clipboard.writeText(text);
      button.textContent = "Copied";
    } catch {
      button.textContent = "Copy failed";
    }
    setTimeout(() => { button.textContent = label; }, 1500);
  });
  return button;
}

function appendProse(parent, text) {
  const clean = text.trim();
  if (!clean) return;
  const prose = element("div", "prose");
  const inline = /\x60([^\x60\n]+)\x60|\*\*([^*\n]+)\*\*/g;
  let offset = 0;
  for (const token of clean.matchAll(inline)) {
    prose.append(clean.slice(offset, token.index));
    prose.append(element(token[1] ? "code" : "strong", "", token[1] ?? token[2]));
    offset = token.index + token[0].length;
  }
  prose.append(clean.slice(offset));
  parent.append(prose);
}

function appendResponse(parent, text) {
  const fences = /^\x60{3}([a-z\d_+.-]*)[ \t]*\r?\n([\s\S]*?)(?:^\x60{3}[ \t]*\r?$|(?![\s\S]))/gim;
  let offset = 0;
  for (const fence of text.matchAll(fences)) {
    appendProse(parent, text.slice(offset, fence.index));
    const code = fence[2].replace(/\r?\n$/, "");
    const block = element("div", "code-block");
    const bar = element("header");
    bar.append(element("span", "", fence[1] || "Code"), copyButton(code, "Copy code"));
    const pre = element("pre");
    pre.append(element("code", "", code));
    block.append(bar, pre);
    parent.append(block);
    offset = fence.index + fence[0].length;
  }
  appendProse(parent, text.slice(offset));
}

function makeMessage(role, text, { note = "", pending = false } = {}) {
  const item = element("article", "message " + role + (pending ? " pending" : ""));
  item.append(element("h2", "message-role", role === "user" ? "You" : "Model"));
  const body = element("div", "message-body");

  if (!pending && role === "assistant") {
    appendResponse(body, text);
    const actions = element("div", "message-actions");
    actions.append(copyButton(text, "Copy reply"));
    if (note) actions.append(element("p", "reply-note", note));
    body.append(actions);
  } else {
    body.append(element("p", "prose", text));
  }
  item.append(body);
  return item;
}

async function sendMessage(event) {
  event.preventDefault();
  const text = prompt.value.trim();
  if (!ready || generating || !text) return;

  setError("");
  generating = true;
  prompt.value = "";
  empty.hidden = true;
  updateControls();
  setStatus("Generating…", "busy");

  const user = { role: "user", content: text };
  const thread = [...history, user];
  const userMessage = makeMessage("user", text);
  const pending = makeMessage("assistant", "…", { pending: true });
  messages.append(userMessage, pending);
  transcript.scrollTop = transcript.scrollHeight;
  try {
    const response = await fetch("/api/chat", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ messages: thread.slice(-maxMessages) }),
    });
    const result = await response.json();
    if (!response.ok) throw new Error(result.error || "The request did not finish.");

    const notes = [];
    if (result.finish_reason === "length") notes.push("Reply reached the token limit.");
    if (thread.length > maxMessages || result.omitted_turns > 0) notes.push("Only recent turns were included.");
    const following = transcript.scrollHeight - transcript.scrollTop - transcript.clientHeight < 24;
    const reply = makeMessage("assistant", result.text, { note: notes.join(" ") });
    pending.replaceWith(reply);
    history.push(user, { role: "assistant", content: result.text });
    if (following) {
      if (reply.offsetHeight > transcript.clientHeight) reply.scrollIntoView({ block: "start" });
      else transcript.scrollTop = transcript.scrollHeight;
    }
  } catch (failure) {
    if (failure instanceof TypeError) {
      setStatus("Disconnected", "error");
      setError("The local model server disconnected. Check the launcher terminal and try again.");
    } else {
      setError(failure.message);
    }
    pending.remove();
    userMessage.remove();
    if (!history.length) empty.hidden = false;
    prompt.value = text;
  } finally {
    generating = false;
    if (status.dataset.state === "busy") setStatus("Ready", "ready");
    updateControls();
    prompt.focus({ preventScroll: true });
  }
}

compose.addEventListener("submit", sendMessage);
prompt.addEventListener("input", updateControls);
prompt.addEventListener("keydown", event => {
  if (event.key === "Enter" && !event.shiftKey && !event.isComposing) {
    event.preventDefault();
    if (!send.disabled) compose.requestSubmit();
  }
});

newChat.addEventListener("click", () => {
  history.length = 0;
  messages.replaceChildren();
  empty.hidden = false;
  prompt.value = "";
  setError("");
  updateControls();
  prompt.focus();
});

for (const example of document.querySelectorAll("[data-example]")) {
  example.addEventListener("click", () => {
    prompt.value = example.lastElementChild.textContent;
    updateControls();
    prompt.focus();
  });
}

async function connect() {
  try {
    const response = await fetch("/api/model");
    if (!response.ok) throw new Error();
    const info = await response.json();
    const checkpoint = document.querySelector("#checkpoint");
    checkpoint.textContent = info.checkpoint;
    checkpoint.title = info.checkpoint;
    document.querySelector("#device").textContent = info.device.toUpperCase();
    ready = true;
    setStatus("Ready", "ready");
    updateControls();
  } catch {
    setStatus("Disconnected", "error");
    setError("Could not reach the local model. Check the launcher terminal and refresh this page.");
  }
}

connect();
