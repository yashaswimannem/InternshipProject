document.getElementById("predictionForm").addEventListener("submit", async function (e) {
    e.preventDefault();
  
    // Get form data
    const formData = new FormData(e.target);
    const data = {};
    formData.forEach((value, key) => {
      data[key] = isNaN(value) ? value : parseFloat(value);
    });
  
    try {
      // Send data to the Flask API
      const response = await fetch("http://127.0.0.1:5000/predict", {
        method: "POST",
        headers: {
          "Content-Type": "application/json",
        },
        body: JSON.stringify(data),
      });
  
      // Parse the JSON response
      const result = await response.json();
      document.getElementById("result").innerText = `Predicted Sales: ${result.predicted_sales}`;
    } catch (error) {
      console.error("Error:", error);
      document.getElementById("result").innerText = "Error making prediction. Check console for details.";
    }
  });
  