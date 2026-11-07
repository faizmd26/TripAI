const THREAD_KEY = "travel_thread_id";
const HTML2PDF_SRC = "https://cdnjs.cloudflare.com/ajax/libs/html2pdf.js/0.10.1/html2pdf.bundle.min.js";

let currentThreadId = storageGet(THREAD_KEY);
let latestAnswerMarkdown = "";
let waitingForApproval = false;
let isBusy = false;
let html2pdfPromise = null;

const AGENT_LABELS = {
  flight_agent: "✈️ Flight Agent",
  hotel_agent: "🏨 Hotel Agent",
  weather_agent: "🌦️ Weather Agent",
  budget_agent: "💰 Budget Agent",
  itinerary_agent: "🗓️ Itinerary Agent"
};

const $ = (id) => document.getElementById(id);

function storageGet(key) {
  try {
    return localStorage.getItem(key) || null;
  } catch (e) {
    return null;
  }
}

function storageSet(key, value) {
  try {
    if (value === null) {
      localStorage.removeItem(key);
    } else {
      localStorage.setItem(key, value);
    }
  } catch (e) {
    /* storage unavailable, thread just won't persist */
  }
}

function smoothBehavior() {
  return window.matchMedia("(prefers-reduced-motion: reduce)").matches ? "auto" : "smooth";
}

function setPrompt(text) {
  const input = $("userInput");
  input.value = text;
  input.focus();
}

function setLoading(isLoading, mode = "draft") {
  isBusy = isLoading;

  $("sendBtn").disabled = isLoading;
  $("approveBtn").disabled = isLoading;
  $("reviseBtn").disabled = isLoading;
  $("newTripBtn").disabled = isLoading;

  const showButtonLoader = isLoading && mode === "draft";
  $("btnText").classList.toggle("hidden", showButtonLoader);
  $("btnLoader").classList.toggle("hidden", !showButtonLoader);

  $("loadingText").textContent =
    mode === "draft" ? "Building your draft..." : "Generating your final plan...";
  $("loadingNote").classList.toggle("hidden", !isLoading);

  document.body.setAttribute("aria-busy", String(isLoading));
}

function showError(message) {
  const errorBox = $("errorBox");
  errorBox.textContent = message;
  errorBox.classList.remove("hidden");
  errorBox.scrollIntoView({ behavior: smoothBehavior(), block: "center" });
}

function hideError() {
  const errorBox = $("errorBox");
  errorBox.classList.add("hidden");
  errorBox.textContent = "";
}

/* Markdown from the model is untrusted: strip active content before inserting. */
function sanitizeHtml(html) {
  const doc = new DOMParser().parseFromString(html, "text/html");

  doc.querySelectorAll("script, iframe, object, embed, style, link, meta, form").forEach((node) => node.remove());

  doc.body.querySelectorAll("*").forEach((el) => {
    Array.from(el.attributes).forEach((attr) => {
      const name = attr.name.toLowerCase();
      const value = attr.value.trim().toLowerCase();
      const isUrlAttr = name === "href" || name === "src" || name === "xlink:href";
      if (name.startsWith("on") || (isUrlAttr && value.startsWith("javascript:"))) {
        el.removeAttribute(attr.name);
      }
    });
  });

  doc.body.querySelectorAll("a[href]").forEach((a) => {
    a.setAttribute("target", "_blank");
    a.setAttribute("rel", "noopener noreferrer");
  });

  doc.body.querySelectorAll("table").forEach((table) => {
    const wrap = doc.createElement("div");
    wrap.className = "table-wrap";
    table.parentNode.insertBefore(wrap, table);
    wrap.appendChild(table);
  });

  return doc.body.innerHTML;
}

function renderMarkdown(element, markdown) {
  const text = markdown || "";
  if (typeof marked !== "undefined") {
    element.innerHTML = sanitizeHtml(marked.parse(text));
  } else {
    element.textContent = text;
  }
}

function showWorkflow(data) {
  const chips = $("agentChips");
  const guardrailBadge = $("guardrailBadge");

  $("supervisorReasoning").textContent =
    data.supervisor_reasoning || "Supervisor routing completed.";

  chips.replaceChildren();
  (data.selected_agents || []).forEach((agent) => {
    const chip = document.createElement("span");
    chip.className = "agent-chip";
    chip.textContent = AGENT_LABELS[agent] || agent;
    chips.appendChild(chip);
  });

  if (data.guardrail_allowed === false) {
    guardrailBadge.textContent = "Guardrail blocked";
    guardrailBadge.classList.add("blocked");
  } else {
    guardrailBadge.textContent = "Guardrail passed";
    guardrailBadge.classList.remove("blocked");
  }

  $("workflowSection").classList.remove("hidden");
}

function showResult(answer, threadId, isDraft = false) {
  latestAnswerMarkdown = answer || "";

  renderMarkdown($("resultBox"), latestAnswerMarkdown);
  $("threadInfo").textContent = `Thread ID: ${threadId}`;
  $("resultTitle").textContent = isDraft ? "Draft travel plan" : "Your final AI travel plan";

  const section = $("resultSection");
  section.classList.remove("hidden");
  section.scrollIntoView({ behavior: smoothBehavior(), block: "start" });
}

function showApproval(data) {
  waitingForApproval = true;
  $("approvalRequest").textContent =
    data.approval_request ||
    "Approve the draft or provide feedback before the final plan is generated.";
  $("approvalSection").classList.remove("hidden");
}

function hideApproval() {
  waitingForApproval = false;
  $("approvalSection").classList.add("hidden");
  $("approvalFeedback").value = "";
}

async function postJSON(url, payload, fallbackError) {
  const response = await fetch(url, {
    method: "POST",
    headers: { "Content-Type": "application/json" },
    body: JSON.stringify(payload)
  });

  let data = null;
  try {
    data = await response.json();
  } catch (e) {
    /* non-JSON response, handled below */
  }

  if (!response.ok || !data || !data.success) {
    throw new Error((data && data.error) || fallbackError);
  }
  return data;
}

async function sendMessage() {
  if (isBusy) {
    return;
  }

  hideError();

  if (waitingForApproval) {
    showError("Please approve or revise the current draft before starting another plan.");
    return;
  }

  const input = $("userInput");
  const message = input.value.trim();

  if (!message) {
    showError("Please enter your travel request first.");
    return;
  }

  setLoading(true, "draft");

  try {
    const data = await postJSON(
      "/api/travel",
      { message: message, thread_id: currentThreadId },
      "Something went wrong."
    );

    currentThreadId = data.thread_id;
    storageSet(THREAD_KEY, currentThreadId);

    showWorkflow(data);

    if (data.requires_approval) {
      showResult(data.itinerary || data.answer, data.thread_id, true);
      showApproval(data);
    } else {
      hideApproval();
      showResult(data.answer, data.thread_id, false);
    }
  } catch (error) {
    showError(error.message);
  } finally {
    setLoading(false, "draft");
  }
}

async function submitApproval(approved) {
  if (isBusy) {
    return;
  }

  hideError();

  if (!currentThreadId || !waitingForApproval) {
    showError("There is no draft waiting for approval.");
    return;
  }

  const feedbackInput = $("approvalFeedback");
  const feedback = feedbackInput.value.trim();

  if (!approved && !feedback) {
    showError("Please enter revision feedback before requesting changes.");
    feedbackInput.focus();
    return;
  }

  setLoading(true, "approval");

  try {
    const data = await postJSON(
      "/api/travel/approve",
      { thread_id: currentThreadId, approved: approved, feedback: feedback },
      "Could not resume the travel workflow."
    );

    showWorkflow(data);
    hideApproval();
    showResult(data.answer, data.thread_id, false);
  } catch (error) {
    showError(error.message);
  } finally {
    setLoading(false, "approval");
  }
}

function newTrip() {
  if (isBusy) {
    return;
  }

  currentThreadId = null;
  latestAnswerMarkdown = "";
  storageSet(THREAD_KEY, null);

  hideError();
  hideApproval();
  $("workflowSection").classList.add("hidden");
  $("resultSection").classList.add("hidden");
  $("resultBox").replaceChildren();

  const input = $("userInput");
  input.value = "";
  input.focus();
  window.scrollTo({ top: 0, behavior: smoothBehavior() });
}

function copyResult() {
  const text = $("resultBox").innerText;

  if (!text) {
    return;
  }

  if (!navigator.clipboard) {
    showError("Could not copy result.");
    return;
  }

  navigator.clipboard
    .writeText(text)
    .then(() => {
      const copyBtn = document.querySelector(".copy-btn");
      const oldText = copyBtn.textContent;
      copyBtn.textContent = "Copied";
      setTimeout(() => {
        copyBtn.textContent = oldText;
      }, 1400);
    })
    .catch(() => {
      showError("Could not copy result.");
    });
}

/* html2pdf is large, so it loads only the first time someone downloads. */
function loadHtml2pdf() {
  if (typeof html2pdf !== "undefined") {
    return Promise.resolve();
  }
  if (!html2pdfPromise) {
    html2pdfPromise = new Promise((resolve, reject) => {
      const script = document.createElement("script");
      script.src = HTML2PDF_SRC;
      script.onload = resolve;
      script.onerror = () => {
        html2pdfPromise = null;
        reject(new Error("PDF library failed to load"));
      };
      document.head.appendChild(script);
    });
  }
  return html2pdfPromise;
}

async function downloadPDF() {
  const pdfContent = $("pdfContent");

  if (!latestAnswerMarkdown || !pdfContent) {
    showError("No travel plan available to download.");
    return;
  }

  hideError();

  const downloadBtn = document.querySelector(".download-btn");
  const oldText = downloadBtn.textContent;
  downloadBtn.textContent = "Preparing PDF...";
  downloadBtn.disabled = true;

  const options = {
    margin: 0.5,
    filename: "ai-travel-plan.pdf",
    image: { type: "jpeg", quality: 0.98 },
    html2canvas: { scale: 2, useCORS: true, backgroundColor: "#ffffff" },
    jsPDF: { unit: "in", format: "a4", orientation: "portrait" },
    pagebreak: { mode: ["avoid-all", "css", "legacy"] }
  };

  try {
    await loadHtml2pdf();
    pdfContent.classList.add("exporting");
    await html2pdf().set(options).from(pdfContent).save();
  } catch (error) {
    showError("Could not download PDF.");
  } finally {
    pdfContent.classList.remove("exporting");
    downloadBtn.textContent = oldText;
    downloadBtn.disabled = false;
  }
}

document.addEventListener("keydown", (event) => {
  if ((event.ctrlKey || event.metaKey) && event.key === "Enter" && document.activeElement === $("userInput")) {
    event.preventDefault();
    sendMessage();
  }
});