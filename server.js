require('dotenv').config();
const express = require('express');
const cors = require('cors');
const sharp = require('sharp');
const path = require('path');

const app = express();
const PORT = process.env.PORT || 3000;
const TOKEN = process.env.REPLICATE_API_TOKEN;

app.use(cors());
app.use(express.json({ limit: '35mb' }));
app.use(express.static(process.cwd()));

function clamp(v, min, max) {
  v = Number(v) || min;
  return Math.max(min, Math.min(max, v));
}

function dataUrlToBuffer(dataUrl) {
  const m = String(dataUrl || '').match(/^data:(image\/[\w.+-]+);base64,(.+)$/);
  if (!m) throw new Error('圖片格式錯誤');
  return { mime: m[1], buffer: Buffer.from(m[2], 'base64') };
}

function toDataUrl(buf, mime) {
  return `data:${mime};base64,${buf.toString('base64')}`;
}

async function prepareImage(dataUrl, maxLong = 1500, maxPixels = 1600000) {
  const { buffer } = dataUrlToBuffer(dataUrl);
  const meta = await sharp(buffer, { failOn: 'none' }).metadata();
  if (!meta.width || !meta.height) throw new Error('無法讀取圖片尺寸');

  const pixels = meta.width * meta.height;
  const longEdge = Math.max(meta.width, meta.height);
  const ratio = Math.min(1, maxLong / longEdge, Math.sqrt(maxPixels / pixels));

  if (ratio >= 1) {
    return { dataUrl, width: meta.width, height: meta.height, resized: false };
  }

  const w = Math.round(meta.width * ratio);
  const h = Math.round(meta.height * ratio);
  const out = await sharp(buffer, { failOn: 'none' })
    .resize(w, h, { fit: 'inside', withoutEnlargement: true })
    .jpeg({ quality: 92, mozjpeg: true })
    .toBuffer();

  return {
    dataUrl: toDataUrl(out, 'image/jpeg'),
    width: w,
    height: h,
    resized: true,
    originalWidth: meta.width,
    originalHeight: meta.height
  };
}

function buildPipelineInput({ pipeline, image, scale, faceEnhance }) {
  const safeScale = clamp(scale, 1, 6);

  if (pipeline === 'text') {
    return {
      model: 'topazlabs/image-upscale',
      label: 'Text Pipeline',
      input: {
        image,
        enhance_model: 'Text Refine',
        upscale_factor: `${safeScale}x`,
        face_enhancement: false
      }
    };
  }

  if (pipeline === 'illustration') {
    return {
      model: 'topazlabs/image-upscale',
      label: 'Illustration Pipeline',
      input: {
        image,
        enhance_model: 'CGI',
        upscale_factor: `${safeScale}x`,
        face_enhancement: false
      }
    };
  }

  if (pipeline === 'enhance') {
    return {
      model: 'topazlabs/image-upscale',
      label: 'Enhance Pipeline',
      input: {
        image,
        enhance_model: 'Low Resolution V2',
        upscale_factor: `${safeScale}x`,
        face_enhancement: Boolean(faceEnhance)
      }
    };
  }

  return {
    model: 'topazlabs/image-upscale',
    label: 'Natural Pipeline',
    input: {
      image,
      enhance_model: 'High Fidelity V2',
      upscale_factor: `${safeScale}x`,
      face_enhancement: Boolean(faceEnhance)
    }
  };
}

async function createPrediction(model, input) {
  const r = await fetch('https://api.replicate.com/v1/predictions', {
    method: 'POST',
    headers: {
      Authorization: `Bearer ${TOKEN}`,
      'Content-Type': 'application/json'
    },
    body: JSON.stringify({ version: model, input })
  });

  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || data.error || `Replicate error ${r.status}`);
  return data;
}

async function getPrediction(id) {
  const r = await fetch(`https://api.replicate.com/v1/predictions/${encodeURIComponent(id)}`, {
    headers: { Authorization: `Bearer ${TOKEN}` }
  });

  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || data.error || `Poll error ${r.status}`);
  return data;
}

app.get('/api/health', (_req, res) => {
  res.json({ ok: true, hasToken: Boolean(TOKEN) });
});

app.post('/api/predict', async (req, res) => {
  try {
    if (!TOKEN) return res.status(500).json({ error: '缺少 REPLICATE_API_TOKEN' });

    const {
      imageBase64,
      scale = 4,
      pipeline = 'natural',
      mode = 'natural',
      faceEnhance = true
    } = req.body || {};

    if (!imageBase64) return res.status(400).json({ error: '沒有圖片' });

    const prepared = await prepareImage(imageBase64);
    const selectedPipeline = pipeline || (mode === 'enhance' ? 'enhance' : 'natural');
    const cfg = buildPipelineInput({
      pipeline: selectedPipeline,
      image: prepared.dataUrl,
      scale,
      faceEnhance
    });

    const pred = await createPrediction(cfg.model, cfg.input);

    res.json({
      id: pred.id,
      status: pred.status,
      selected_model_label: cfg.label,
      resized: prepared.resized,
      prepared_size: `${prepared.width}x${prepared.height}`
    });
  } catch (e) {
    const msg = String(e.message || e);
    res.status(500).json({
      error: /cuda|memory|out of memory/i.test(msg)
        ? '模型記憶體不足，請降低倍率或改用 Natural 模式。'
        : '建立任務失敗',
      detail: msg
    });
  }
});

app.get('/api/poll', async (req, res) => {
  try {
    const id = req.query.id;
    if (!id) return res.status(400).json({ error: '缺少 id' });
    res.json(await getPrediction(id));
  } catch (e) {
    res.status(500).json({ error: '查詢失敗', detail: String(e.message || e) });
  }
});

app.get('*', (_req, res) => {
  res.sendFile(path.join(process.cwd(), 'index.html'));
});

app.listen(PORT, () => console.log('Server running:', PORT));