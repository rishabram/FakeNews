document.getElementById('checkBtn').addEventListener('click', async () => {
  let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });
  const results = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: () => window.getSelection().toString()
  });

  const selectedText = results[0].result || "";
  if (!selectedText) {
    document.getElementById('result').textContent = "No text selected!";
    return;
  }

  const flaskUrl = "http://127.0.0.1:5000/predict";

  try {
    const response = await fetch(flaskUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: selectedText })
    });
    const data = await response.json();
    document.getElementById('result').textContent =
      `Prediction: ${data.prediction}, Confidence: ${data.confidence_label}`;
  } catch (error) {
    console.error(error);
    document.getElementById('result').textContent = "Error contacting the server.";
  }
});
