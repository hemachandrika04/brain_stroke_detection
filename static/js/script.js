document.getElementById("uploadForm").addEventListener("submit", async function (e) {
    e.preventDefault();

    const file = document.getElementById("imageInput").files[0];
    const preview = document.getElementById("previewImage");
    const details = document.getElementById("imageDetails");
    const output = document.getElementById("output");

    if (!file) {
        output.textContent = "Please select a brain scan image.";
        return;
    }

    const formData = new FormData();
    formData.append("image", file);

    output.className = "diagnosis";
    output.textContent = "Running AI analysis...";
    preview.hidden = true;
    details.innerHTML = "";

    const response = await fetch("/predict", {
        method: "POST",
        body: formData
    });

    const data = await response.json();

    preview.src = "/" + data.image_path;
    preview.hidden = false;

    details.innerHTML = `
        <strong>Input Image</strong><br>
        File: ${data.file_name}<br>
        Size: ${data.file_size} KB<br>
        Resolution: ${data.dimensions}
    `;

    output.className = "diagnosis";
    if (data.result === "Stroke Detected") {
        output.classList.add("result-red");
    } else {
        output.classList.add("result-green");
    }

    output.innerHTML = `
        ${data.result}<br>
        Confidence: ${(data.confidence * 100).toFixed(2)}%
    `;
});
