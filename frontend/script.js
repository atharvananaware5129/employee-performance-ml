const BASE_URL = "http://127.0.0.1:8000"; // your FastAPI backend

// Train Model
async function trainModel() {
    const file = document.getElementById("trainFile").files[0];
    const target = document.getElementById("trainTarget").value;

    if (!file) {
        alert("Please select a CSV file to train.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);
    if (target) formData.append("target_col", target);

    const res = await fetch(`${BASE_URL}/train`, {
        method: "POST",
        body: formData
    });

    const result = await res.json();
    document.getElementById("trainResult").innerText = result.message || result.detail;
}

// Test Model
async function testModel() {
    const file = document.getElementById("testFile").files[0];

    if (!file) {
        alert("Please select a CSV file to test.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch(`${BASE_URL}/test`, {
        method: "POST",
        body: formData
    });

    const result = await res.json();
    if (res.ok) {
        document.getElementById("testResult").innerHTML = `
            Accuracy: ${result.accuracy}<br>
            Precision: ${result.precision}<br>
            Recall: ${result.recall}<br>
            F1 Score: ${result.f1_score}
        `;
    } else {
        document.getElementById("testResult").innerText = result.detail;
    }
}

// Predict Model
async function predictModel() {
    const file = document.getElementById("predictFile").files[0];

    if (!file) {
        alert("Please select a CSV file to predict.");
        return;
    }

    const formData = new FormData();
    formData.append("file", file);

    const res = await fetch(`${BASE_URL}/predict`, {
        method: "POST",
        body: formData
    });

    const result = await res.json();
    if (res.ok) {
        document.getElementById("predictResult").innerText = JSON.stringify(result.predictions, null, 2);
    } else {
        document.getElementById("predictResult").innerText = result.detail;
    }
}
