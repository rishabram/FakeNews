document.getElementById('checkBtn').addEventListener('click', async () => {
  // 1. Get the selected text from the current tab’s DOM
  let [tab] = await chrome.tabs.query({ active: true, currentWindow: true });

  // Run a script in the page to get the selection
  const results = await chrome.scripting.executeScript({
    target: { tabId: tab.id },
    func: () => window.getSelection().toString()  // returns currently highlighted text
  });

  const selectedText = results[0].result || "";
  if (!selectedText) {
    document.getElementById('result').textContent = "No text selected!";
    return;
  }

  // 2. Call the Flask API with the selected text
  const flaskUrl = "http://127.0.0.1:5000/predict";

  try {
    const response = await fetch(flaskUrl, {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: selectedText })
    });
    const data = await response.json();

    // 3. Show the result in the popup
    document.getElementById('result').textContent =
      `Prediction: ${data.prediction}, Confidence: ${data.confidence}`;
  } catch (error) {
    console.error(error);
    document.getElementById('result').textContent = "Error contacting the server.";
  }
});