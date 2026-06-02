let session = null;
let video = document.getElementById('webcam');
let focusBox = document.getElementById('focus-box');
let percentText = document.getElementById('percent-text');
let rawText = document.getElementById('raw-text');
let hintText = document.getElementById('hint-text');
let zoomSlider = document.getElementById('zoom-slider');
let onnxFileInput = document.getElementById('onnx-file-input');

// Variables de configuración dinámica del modelo
let modelWidth = 224;  // Valor por defecto (se auto-ajustará al cargar el modelo si es posible)
let modelHeight = 224; // Valor por defecto
let modelChannels = 3;  // Valor por defecto
let inputIsNHWC = false; // Indica si el modelo espera [batch, alto, ancho, canales] en lugar de [batch, canales, alto, ancho]

// Variable global para almacenar la imagen estática cargada
let currentUploadedImage = null;
let currentFileSizeText = "---"; // Guardará el peso del archivo leído

// Ratio de tamaño del recuadro central respecto al contenedor total (70%)
const boxRatio = 0.7;

// Ajustamos programáticamente las propiedades de la UI para mayor dinamismo y precisión
window.addEventListener('DOMContentLoaded', () => {
    if (zoomSlider) {
        zoomSlider.min = "0.5"; // Permitir zoom inverso (alejar hasta 0.5x)
        zoomSlider.max = "3.0"; // Acercar hasta 3x
        zoomSlider.step = "0.05"; // Paso de ajuste ultra-fino
        zoomSlider.value = "1.0"; // Valor de inicio neutro
    }
    
    if (focusBox) {
        focusBox.style.top = "15%";
        focusBox.style.left = "15%";
        focusBox.style.width = "70%";
        focusBox.style.height = "70%";
    }
});

// Escuchar cambios en el slider de zoom para aplicar efectos visuales y re-procesar
zoomSlider.addEventListener('input', () => {
    const zoom = parseFloat(zoomSlider.value);
    
    video.style.transform = `scale(${zoom})`;
    video.style.transformOrigin = 'center';

    let previewImg = document.getElementById('preview-img');
    if (previewImg) {
        previewImg.style.transform = `scale(${zoom})`;
        previewImg.style.transformOrigin = 'center';
    }

    // Si hay una imagen cargada, re-ejecutamos la IA sobre el nuevo encuadre
    if (currentUploadedImage) {
        processStaticImage();
    }
});

// 1. CARGA SEGURA DEL MODELO ONNX Y DIAGNÓSTICO DE PARÁMETROS
document.getElementById('btn-load-model').addEventListener('click', () => {
    onnxFileInput.click();
});

onnxFileInput.addEventListener('change', async (e) => {
    const file = e.target.files[0];
    if (!file) return;

    const statusText = document.getElementById('status-text');
    statusText.innerText = "Cargando modelo en memoria... por favor espera.";
    
    console.log("=== INICIO DE DEPURACIÓN DE CARGA DEL MODELO ===");
    console.log("Archivo seleccionado:", file.name);
    console.log("Tamaño del archivo:", (file.size / (1024 * 1024)).toFixed(2), "MB");

    try {
        const arrayBuffer = await file.arrayBuffer();
        session = await ort.InferenceSession.create(new Uint8Array(arrayBuffer));
        
        // --- PROCESO DE AUTODETECCIÓN DE PARÁMETROS CON ACCESO ULTRA SEGURO ---
        const inputName = (session.inputNames && session.inputNames.length > 0) ? session.inputNames[0] : "input";
        const outputName = (session.outputNames && session.outputNames.length > 0) ? session.outputNames[0] : "output";
        
        let inputInfo = null;
        if (session.inputs) {
            if (session.inputs[inputName]) {
                inputInfo = session.inputs[inputName];
            } else if (Array.isArray(session.inputs)) {
                inputInfo = session.inputs.find(i => i && (i.name === inputName));
            } else {
                const keys = Object.keys(session.inputs);
                if (keys.length > 0) {
                    inputInfo = session.inputs[keys[0]];
                }
            }
        }
        
        console.log("Modelo ONNX cargado exitosamente.");
        console.log("Nombre del nodo de entrada:", inputName);
        console.log("Nombre del nodo de salida:", outputName);
        
        if (inputInfo && inputInfo.dims) {
            console.log("Dimensiones de entrada esperadas por el modelo:", inputInfo.dims);
            
            const dims = inputInfo.dims;
            if (dims.length === 4) {
                if (dims[1] === 3 || dims[1] === 1) {
                    modelChannels = dims[1];
                    modelHeight = dims[2];
                    modelWidth = dims[3];
                    inputIsNHWC = false;
                } else if (dims[3] === 3 || dims[3] === 1) {
                    modelHeight = dims[1];
                    modelWidth = dims[2];
                    modelChannels = dims[3];
                    inputIsNHWC = true;
                }
            }
        }
        
        statusText.innerText = `¡IA Cargada con éxito! (${modelWidth}x${modelHeight})`;
        document.getElementById('setup-container').style.display = 'none';
        document.getElementById('main-interface').style.display = 'block';
        
        startCamera();
    } catch (err) {
        console.error("Fallo al inicializar el modelo ONNX:", err);
        statusText.innerText = `Error al cargar: ${err.message || err}`;
    }
});

// 2. ACTIVACIÓN DE LA CÁMARA
async function startCamera() {
    try {
        // Inicializar metadatos para la cámara
        document.getElementById('meta-size').innerText = "Streaming (En vivo)";
        
        const stream = await navigator.mediaDevices.getUserMedia({
            video: { facingMode: "environment", width: 640, height: 480 },
            audio: false
        });
        video.srcObject = stream;
        video.addEventListener('loadeddata', () => {
            // Exponer metadatos de cámara
            document.getElementById('meta-resolution').innerText = `${video.videoWidth} x ${video.videoHeight}`;
            document.getElementById('meta-aspect').innerText = getAspectRatioText(video.videoWidth, video.videoHeight);
            predictLoop();
        });
    } catch (err) {
        console.error("Error al acceder a la cámara: ", err);
        hintText.innerText = "Hint: No se pudo acceder a la cámara. Intenta cargar una imagen manual.";
    }
}

// 3. BUCLE DE INFERENCIA EN TIEMPO REAL
async function predictLoop() {
    if (!session || video.paused || video.ended) {
        requestAnimationFrame(predictLoop);
        return;
    }

    await processVideoFrame();
    requestAnimationFrame(predictLoop);
}

// 4. PROCESAMIENTO DE IMAGEN PARA LA IA (CÁMARA EN VIVO)
async function processVideoFrame() {
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    
    canvas.width = modelWidth;
    canvas.height = modelHeight;

    const zoom = parseFloat(zoomSlider.value);
    
    let cropWidth = (video.videoWidth * boxRatio) / zoom;
    let cropHeight = (video.videoHeight * boxRatio) / zoom;
    
    let sX = (video.videoWidth - cropWidth) / 2;
    let sY = (video.videoHeight - cropHeight) / 2;
    
    let dX = 0;
    let dY = 0;
    let dW = modelWidth;
    let dH = modelHeight;

    ctx.fillStyle = '#131a35';
    ctx.fillRect(0, 0, modelWidth, modelHeight);

    if (sX < 0 || sY < 0) {
        const scale = zoom / boxRatio;
        dW = modelWidth * scale;
        dH = modelHeight * scale;
        dX = (modelWidth - dW) / 2;
        dY = (modelHeight - dH) / 2;
        
        sX = 0;
        sY = 0;
        cropWidth = video.videoWidth;
        cropHeight = video.videoHeight;
    }

    ctx.drawImage(video, sX, sY, cropWidth, cropHeight, dX, dY, dW, dH);
    
    const imgData = ctx.getImageData(0, 0, modelWidth, modelHeight);

    // Calcular color promedio del enfoque
    updateAverageColor(imgData);

    // Medición de enfoque (Laplacian Sharpness)
    const sharpness = calculateSharpness(ctx, modelWidth, modelHeight);

    try {
        const results = await executeInferenceWithRecovery(imgData);
        const outputName = session.outputNames[0];
        const output = results[outputName].data;
        const confidence = output[0]; 
        
        // Actualizar UI con resultados de IA
        updateUI(confidence);
        
        // Calcular e ilustrar compatibilidad de similitud
        calculateCompatibility(confidence, sharpness, zoom);
    } catch (err) {
        console.debug("Inferencia en vivo en espera:", err);
    }
}

// Convierte ImageData a Float32 en formato Planar NCHW o NHWC
function imageDataToTensor(imgData, width, height) {
    const floatData = new Float32Array(width * height * modelChannels);
    
    if (inputIsNHWC) {
        for (let i = 0; i < imgData.data.length / 4; i++) {
            floatData[i * 3] = imgData.data[i * 4] / 255.0;         // R
            floatData[i * 3 + 1] = imgData.data[i * 4 + 1] / 255.0; // G
            floatData[i * 3 + 2] = imgData.data[i * 4 + 2] / 255.0; // B
        }
    } else {
        for (let i = 0; i < imgData.data.length / 4; i++) {
            floatData[i] = imgData.data[i * 4] / 255.0;                         // R
            floatData[i + width * height] = imgData.data[i * 4 + 1] / 255.0;     // G
            floatData[i + width * height * 2] = imgData.data[i * 4 + 2] / 255.0; // B
        }
    }
    
    const shape = inputIsNHWC ? [1, height, width, modelChannels] : [1, modelChannels, height, width];
    return new ort.Tensor('float32', floatData, shape);
}

// 5. MOTOR DE INFERENCIA RESILIENTE (AUTO-RECUPERACIÓN DE DIMENSIONES)
async function executeInferenceWithRecovery(imgData) {
    const inputName = session.inputNames[0];
    let inputTensor = imageDataToTensor(imgData, modelWidth, modelHeight);
    
    try {
        return await session.run({ [inputName]: inputTensor });
    } catch (err) {
        const errMsg = err.message || "";
        if (errMsg.includes("dimensions") || errMsg.includes("dimension mismatch") || errMsg.includes("OrtRun")) {
            console.warn("⚠️ ¡Conflicto de dimensiones detectado en ONNX Runtime! Ejecutando auto-recuperación activa...");
            inputIsNHWC = !inputIsNHWC;
            
            inputTensor = imageDataToTensor(imgData, modelWidth, modelHeight);
            return await session.run({ [inputName]: inputTensor });
        }
        throw err;
    }
}

// 6. ACTUALIZACIÓN DE LA UI
function updateUI(confidence) {
    const boundedConfidence = Math.max(0, Math.min(1, confidence));
    const percent = (boundedConfidence * 100).toFixed(2);
    
    percentText.innerText = `${percent}%`;
    rawText.innerText = boundedConfidence.toFixed(4);

    if (boundedConfidence >= 0.7000) {
        focusBox.style.borderColor = "#00FF00";
        focusBox.style.boxShadow = "0 0 15px #00FF00";
        hintText.innerText = "Lectura Correcta: Objeto Verificado.";
        hintText.style.color = "#00FF00";
    } else {
        focusBox.style.borderColor = "#FF0000";
        focusBox.style.boxShadow = "0 0 15px #FF0000";
        hintText.style.color = "#FF3333";

        if (boundedConfidence >= 0.1500 && boundedConfidence < 0.6000) {
            hintText.innerText = "Hint: Ajusta el zoom para que la taza completa esté en el recuadro verde.";
        } else if (boundedConfidence > 0.0500 && boundedConfidence < 0.1500) {
            hintText.innerText = "Hint: Aleje un poco la imagen con la barra de zoom.";
        } else {
            hintText.innerText = "Hint: Coloque la taza en el recuadro central o aleje la escala.";
        }
    }
}

// 7. CARGAR IMAGEN DESDE PC
const fileInput = document.getElementById('file-input');
document.getElementById('btn-upload-img').addEventListener('click', () => fileInput.click());

fileInput.addEventListener('change', (e) => {
    const file = e.target.files[0];
    if (!file) return;

    // Calcular y exponer el peso del archivo
    if (file.size > 1024 * 1024) {
        currentFileSizeText = `${(file.size / (1024 * 1024)).toFixed(2)} MB`;
    } else {
        currentFileSizeText = `${(file.size / 1024).toFixed(1)} KB`;
    }
    document.getElementById('meta-size').innerText = currentFileSizeText;

    // Pausar cámara si está activa
    if (video.srcObject) {
        const tracks = video.srcObject.getTracks();
        tracks.forEach(track => track.stop());
    }

    const reader = new FileReader();
    reader.onload = function(event) {
        const img = new Image();
        img.onload = function() {
            // Exponer metadatos de resolución de imagen cargada
            document.getElementById('meta-resolution').innerText = `${img.width} x ${img.height}`;
            document.getElementById('meta-aspect').innerText = getAspectRatioText(img.width, img.height);
            
            currentUploadedImage = img;
            zoomSlider.value = "1.0";

            processStaticImage();
        };
        img.src = event.target.result;
    };
    reader.readAsDataURL(file);
    document.getElementById('btn-restart-cam').style.display = 'inline-block';
});

// Procesar, recortar, previsualizar y analizar la imagen estática
function processStaticImage() {
    if (!currentUploadedImage) return;

    const zoom = parseFloat(zoomSlider.value);
    
    let previewImg = document.getElementById('preview-img');
    if (!previewImg) {
        previewImg = document.createElement('img');
        previewImg.id = 'preview-img';
        previewImg.style.width = '100%';
        previewImg.style.height = 'auto'; 
        previewImg.style.display = 'block'; 
        previewImg.style.backgroundColor = '#131a35';
        previewImg.style.transition = 'transform 0.1s ease-out';
        video.parentNode.insertBefore(previewImg, video);
    }
    
    previewImg.src = currentUploadedImage.src;
    previewImg.style.display = 'block';
    
    previewImg.style.transform = `scale(${zoom})`;
    previewImg.style.transformOrigin = 'center';
    
    video.style.display = 'none';

    // Recorte del Canvas para enviar a la IA
    const canvas = document.createElement('canvas');
    const ctx = canvas.getContext('2d');
    canvas.width = modelWidth;
    canvas.height = modelHeight;

    let cropWidth = (currentUploadedImage.width * boxRatio) / zoom;
    let cropHeight = (currentUploadedImage.height * boxRatio) / zoom;
    
    let sX = (currentUploadedImage.width - cropWidth) / 2;
    let sY = (currentUploadedImage.height - cropHeight) / 2;
    
    let dX = 0;
    let dY = 0;
    let dW = modelWidth;
    let dH = modelHeight;

    ctx.fillStyle = '#131a35';
    ctx.fillRect(0, 0, modelWidth, modelHeight);

    if (sX < 0 || sY < 0) {
        const scale = zoom / boxRatio;
        dW = modelWidth * scale;
        dH = modelHeight * scale;
        dX = (modelWidth - dW) / 2;
        dY = (modelHeight - dH) / 2;
        
        sX = 0;
        sY = 0;
        cropWidth = currentUploadedImage.width;
        cropHeight = currentUploadedImage.height;
    }

    ctx.drawImage(currentUploadedImage, sX, sY, cropWidth, cropHeight, dX, dY, dW, dH);

    const imgData = ctx.getImageData(0, 0, modelWidth, modelHeight);
    
    // Analizar color y nitidez de la imagen estática
    updateAverageColor(imgData);
    const sharpness = calculateSharpness(ctx, modelWidth, modelHeight);

    if (!session) {
        console.error("Error: Sesión de ONNX no inicializada.");
        hintText.innerText = "Error: Carga primero el archivo del modelo ONNX.";
        return;
    }

    executeInferenceWithRecovery(imgData).then(results => {
        const outputName = session.outputNames[0];
        const rawOutput = results[outputName].data;
        const confidence = rawOutput[0];
        
        updateUI(confidence);
        calculateCompatibility(confidence, sharpness, zoom);
    }).catch(err => {
        console.error("Error en inferencia de imagen estática:", err);
        hintText.innerText = `Error: Falló la inferencia de la imagen. Detalles: ${err.message || err}`;
    });
}

// Reactivar el flujo de la cámara en vivo
document.getElementById('btn-restart-cam').addEventListener('click', () => {
    document.getElementById('btn-restart-cam').style.display = 'none';
    
    currentUploadedImage = null;
    zoomSlider.value = "1.0";
    
    video.style.transform = 'scale(1)';
    
    let previewImg = document.getElementById('preview-img');
    if (previewImg) {
        previewImg.style.display = 'none';
    }
    video.style.display = 'block';
    
    startCamera();
});

// --- FUNCIONES AUXILIARES DE ANÁLISIS ---

// Obtener formato legible de la relación de aspecto
function getAspectRatioText(width, height) {
    const ratio = width / height;
    if (Math.abs(ratio - 1) < 0.15) return "1:1 Cuadrado";
    if (Math.abs(ratio - 1.33) < 0.15) return "4:3 Estándar";
    if (Math.abs(ratio - 1.77) < 0.15) return "16:9 Panorámica";
    return `${ratio.toFixed(2)}:1`;
}

// Calcular y renderizar el color promedio del recorte analizado
function updateAverageColor(imgData) {
    let rSum = 0, gSum = 0, bSum = 0;
    const totalPixels = imgData.width * imgData.height;
    
    for (let i = 0; i < imgData.data.length; i += 4) {
        rSum += imgData.data[i];
        gSum += imgData.data[i + 1];
        bSum += imgData.data[i + 2];
    }
    
    const rAvg = Math.round(rSum / totalPixels);
    const gAvg = Math.round(gSum / totalPixels);
    const bAvg = Math.round(bSum / totalPixels);
    
    const rgbText = `RGB(${rAvg}, ${gAvg}, ${bAvg})`;
    document.getElementById('meta-color').innerText = rgbText;
    document.getElementById('color-chip').style.backgroundColor = `rgb(${rAvg}, ${gAvg}, ${bAvg})`;
}

// Medidor de nitidez basado en gradiente básico
function calculateSharpness(ctx, w, h) {
    const imgData = ctx.getImageData(0, 0, w, h);
    const d = imgData.data;
    let totalDiff = 0;
    let samples = 0;

    for (let y = 10; y < h - 10; y += 4) {
        for (let x = 10; x < w - 10; x += 4) {
            const idx = (y * w + x) * 4;
            const val = (d[idx] + d[idx+1] + d[idx+2]) / 3;
            const rightVal = (d[idx + 4] + d[idx + 5] + d[idx + 6]) / 3;
            totalDiff += Math.abs(val - rightVal);
            samples++;
        }
    }
    return totalDiff / samples;
}

// Calcular coeficientes de compatibilidad y similitud respecto a la estructura de la taza
function calculateCompatibility(confidence, sharpness, zoom) {
    // 1. Alineación de Escala: El modelo fue entrenado con una taza cuadrada completa.
    // Un zoom óptimo cercano a 1.0 (o 0.8 en imágenes de cuerpo entero) es ideal.
    let aspectScore = 100 - Math.min(100, Math.abs(1.0 - zoom) * 40);
    aspectScore = Math.max(10, Math.min(100, Math.round(aspectScore)));
    
    // 2. Foco / Nitidez: Los modelos esperan imágenes limpias y no borrosas.
    let focusScore = (sharpness / 20) * 100;
    focusScore = Math.max(15, Math.min(100, Math.round(focusScore)));

    // 3. Similitud Estructural Total (Pattern Similarity):
    // Combina la confianza predictiva de la IA (80%), alineación de escala (10%) y nitidez (10%)
    const rawConfidence = Math.max(0, Math.min(1, confidence));
    let patternScore = (rawConfidence * 80) + (aspectScore * 0.10) + (focusScore * 0.10);
    patternScore = Math.round(patternScore);

    // Actualizar barras de progreso y textos en la UI
    document.getElementById('val-aspect-match').innerText = `${aspectScore}%`;
    document.getElementById('fill-aspect-match').style.width = `${aspectScore}%`;

    document.getElementById('val-focus-match').innerText = `${focusScore}%`;
    document.getElementById('fill-focus-match').style.width = `${focusScore}%`;

    document.getElementById('val-pattern-similarity').innerText = `${patternScore}%`;
    document.getElementById('fill-pattern-similarity').style.width = `${patternScore}%`;
}