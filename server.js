require('dotenv').config();
const express = require('express');
const cors = require('cors');
const sharp = require('sharp');
const path = require('path');
const crypto = require('crypto');

const app = express();
const PORT = process.env.PORT || 3000;
const TOKEN = process.env.REPLICATE_API_TOKEN;

const ENABLE_SWINIR_DEFAULT = String(process.env.ENABLE_SWINIR || 'true').toLowerCase() !== 'false';
const ENABLE_GFP_DEFAULT = String(process.env.ENABLE_GFP || 'true').toLowerCase() !== 'false';

app.use(cors());
app.use(express.json({ limit: '40mb' }));
app.use(express.static(process.cwd()));

const jobs = new Map();

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

function outputUrl(output) {
  if (Array.isArray(output)) return output[0];
  return output;
}

async function prepareImage(dataUrl, maxLong = 1500, maxPixels = 1600000) {
  const { mime, buffer } = dataUrlToBuffer(dataUrl);
  const meta = await sharp(buffer, { failOn: 'none' }).metadata();
  if (!meta.width || !meta.height) throw new Error('無法讀取圖片尺寸');

  const pixels = meta.width * meta.height;
  const longEdge = Math.max(meta.width, meta.height);
  const ratio = Math.min(1, maxLong / longEdge, Math.sqrt(maxPixels / pixels));

  if (ratio >= 1) {
    return { dataUrl, mime, width: meta.width, height: meta.height, resized: false };
  }

  const w = Math.round(meta.width * ratio);
  const h = Math.round(meta.height * ratio);
  const hasAlpha = Boolean(meta.hasAlpha) || mime === 'image/png';

  let pipeline = sharp(buffer, { failOn: 'none' })
    .resize(w, h, { fit: 'inside', withoutEnlargement: true });

  let outMime = 'image/jpeg';
  if (hasAlpha) {
    pipeline = pipeline.png({ compressionLevel: 9 });
    outMime = 'image/png';
  } else {
    pipeline = pipeline.jpeg({ quality: 92, mozjpeg: true });
  }

  const out = await pipeline.toBuffer();

  return {
    dataUrl: toDataUrl(out, outMime),
    mime: outMime,
    width: w,
    height: h,
    resized: true,
    originalWidth: meta.width,
    originalHeight: meta.height
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
  if (!r.ok) throw new Error(data.detail || data.error || `Replicate create error ${r.status}`);
  return data;
}

async function getPrediction(id) {
  const r = await fetch(`https://api.replicate.com/v1/predictions/${encodeURIComponent(id)}`, {
    headers: { Authorization: `Bearer ${TOKEN}` }
  });

  const data = await r.json().catch(() => ({}));
  if (!r.ok) throw new Error(data.detail || data.error || `Replicate poll error ${r.status}`);
  return data;
}

async function runReplicate(model, input, onProgress, label) {
  const pred = await createPrediction(model, input);
  let current = pred;

  for (let i = 0; i < 180; i++) {
    current = await getPrediction(pred.id);

    if (current.status === 'succeeded') {
      const out = outputUrl(current.output);
      if (!out) throw new Error(`${label} 沒有輸出圖片`);
      return out;
    }

    if (current.status === 'failed' || current.status === 'canceled') {
      throw new Error(current.error || `${label} 處理失敗`);
    }

    if (onProgress) onProgress(label, current.status);
    await new Promise(resolve => setTimeout(resolve, 2200));
  }

  throw new Error(`${label} 等待逾時`);
}

function buildTopazInput({ image, scale, model, faceEnhance }) {
  const safeScale = clamp(scale, 1, 6);
  return {
    image,
    enhance_model: model,
    upscale_factor: `${safeScale}x`,
    face_enhancement: Boolean(faceEnhance)
  };
}

async function runPipeline(job, payload) {
  try {
    if (!TOKEN) throw new Error('缺少 REPLICATE_API_TOKEN');

    const {
      imageBase64,
      scale = 4,
      pipeline = 'photogrid',
      faceEnhance = true,
      secondStage = true
    } = payload || {};

    if (!imageBase64) throw new Error('沒有圖片');

    job.status = 'processing';
    job.progress = 8;
    job.message = '前處理與大圖保護中…';

    const prepared = await prepareImage(imageBase64);
    job.prepared_size = `${prepared.width}x${prepared.height}`;
    job.resized = prepared.resized;

    let image = prepared.dataUrl;
    const useSwinIR = ENABLE_SWINIR_DEFAULT && Boolean(secondStage);
    const useGFPGAN = ENABLE_GFP_DEFAULT && Boolean(faceEnhance);

    if (pipeline === 'text') {
      job.progress = 18;
      job.message = 'Text Refine 文字銳化中…';
      image = await runReplicate(
        'topazlabs/image-upscale',
        buildTopazInput({ image, scale, model: 'Text Refine', faceEnhance: false }),
        () => {},
        'Text Refine'
      );
    } else if (pipeline === 'illustration') {
      job.progress = 18;
      job.message = 'CGI 插畫邊緣優化中…';
      image = await runReplicate(
        'topazlabs/image-upscale',
        buildTopazInput({ image, scale, model: 'CGI', faceEnhance: false }),
        () => {},
        'CGI'
      );

      if (useSwinIR) {
        job.progress = 66;
        job.message = 'SwinIR 第二層細節補強中…';
        image = await runReplicate(
          'jingyunliang/swinir',
          { image, task_type: 'Real-World Image Super-Resolution-Large', noise: 15, jpeg: 40 },
          () => {},
          'SwinIR'
        );
      }
    } else if (pipeline === 'restore') {
      job.progress = 18;
      job.message = 'Low Resolution V2 強化修復中…';
      image = await runReplicate(
        'topazlabs/image-upscale',
        buildTopazInput({ image, scale: Math.min(Number(scale) || 4, 4), model: 'Low Resolution V2', faceEnhance }),
        () => {},
        'Low Resolution V2'
      );

      if (useGFPGAN) {
        job.progress = 52;
        job.message = 'GFPGAN 臉部修復中…';
        image = await runReplicate(
          'tencentarc/gfpgan',
          { img: image, version: 'v1.4', scale: 1, weight: 0.5 },
          () => {},
          'GFPGAN'
        );
      }

      if (useSwinIR) {
        job.progress = 76;
        job.message = 'SwinIR 第二層細節重建中…';
        image = await runReplicate(
          'jingyunliang/swinir',
          { image, task_type: 'Real-World Image Super-Resolution-Large', noise: 15, jpeg: 40 },
          () => {},
          'SwinIR'
        );
      }
    } else {
      job.progress = 18;
      job.message = 'Topaz High Fidelity 主模型處理中…';
      image = await runReplicate(
        'topazlabs/image-upscale',
        buildTopazInput({ image, scale, model: 'High Fidelity V2', faceEnhance }),
        () => {},
        'High Fidelity V2'
      );

      if (useSwinIR) {
        job.progress = 70;
        job.message = 'SwinIR 第二層細節補強中…';
        image = await runReplicate(
          'jingyunliang/swinir',
          { image, task_type: 'Real-World Image Super-Resolution-Large', noise: 15, jpeg: 40 },
          () => {},
          'SwinIR'
        );
      }
    }

    job.status = 'succeeded';
    job.progress = 100;
    job.message = '完成';
    job.output = image;
    job.finished_at = Date.now();
  } catch (e) {
    const msg = String(e.message || e);
    job.status = 'failed';
    job.error = /cuda|memory|out of memory/i.test(msg)
      ? '模型記憶體不足，請降低倍率、關閉第二層 SwinIR 或改用 Photogrid Pro 模式。'
      : msg;
    job.detail = msg;
    job.finished_at = Date.now();
  }
}

app.get('/api/health', (_req, res) => {
  res.json({ ok: true, hasToken: Boolean(TOKEN), swinir: ENABLE_SWINIR_DEFAULT, gfpgan: ENABLE_GFP_DEFAULT });
});

app.post('/api/predict', async (req, res) => {
  const id = crypto.randomUUID();
  const job = {
    id,
    status: 'starting',
    progress: 1,
    message: '任務建立中…',
    created_at: Date.now()
  };
  jobs.set(id, job);
  runPipeline(job, req.body);
  res.json({ id, status: job.status });
});

app.get('/api/poll', (req, res) => {
  const id = String(req.query.id || '');
  const job = jobs.get(id);
  if (!job) return res.status(404).json({ error: '找不到任務' });
  res.json(job);
});

async function loadImageBufferFromUrlOrDataUrl(inputUrl) {
  const value = String(inputUrl || '');

  if (value.startsWith('data:image/')) {
    const { buffer, mime } = dataUrlToBuffer(value);
    return { buffer, mime };
  }

  if (!/^https?:\/\//i.test(value)) {
    throw new Error('圖片 URL 格式不正確');
  }

  const r = await fetch(value, {
    redirect: 'follow',
    headers: {
      'User-Agent': 'Mozilla/5.0 UrbanUpscaleDownloader/1.0',
      'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8'
    }
  });

  if (!r.ok) {
    throw new Error(`無法下載模型輸出圖片：HTTP ${r.status}`);
  }

  const contentType = r.headers.get('content-type') || '';
  const arrayBuffer = await r.arrayBuffer();
  const buffer = Buffer.from(arrayBuffer);

  if (!buffer.length) {
    throw new Error('模型輸出圖片為空');
  }

  return { buffer, mime: contentType.split(';')[0] || 'application/octet-stream' };
}

async function loadDownloadBuffer(inputUrl) {
  const value = String(inputUrl || '');

  if (value.startsWith('data:image/')) {
    const { buffer, mime } = dataUrlToBuffer(value);
    return { buffer, mime };
  }

  if (!/^https?:\/\//i.test(value)) {
    throw new Error('圖片 URL 格式不正確');
  }

  const r = await fetch(value, {
    redirect: 'follow',
    headers: {
      'User-Agent': 'Mozilla/5.0 UrbanUpscaleDownloader/1.0',
      'Accept': 'image/avif,image/webp,image/apng,image/svg+xml,image/*,*/*;q=0.8'
    }
  });

  if (!r.ok) {
    throw new Error(`無法下載模型輸出圖片：HTTP ${r.status}`);
  }

  const arrayBuffer = await r.arrayBuffer();
  const buffer = Buffer.from(arrayBuffer);
  if (!buffer.length) throw new Error('模型輸出圖片為空');

  return { buffer, mime: r.headers.get('content-type') || 'application/octet-stream' };
}

app.post('/api/download', async (req, res) => {
  try {
    const { url, format = 'png' } = req.body || {};
    if (!url) return res.status(400).json({ error: '缺少圖片 URL' });

    const { buffer } = await loadDownloadBuffer(url);
    const fmt = String(format || 'png').toLowerCase();

    let pipeline = sharp(buffer, { failOn: 'none', animated: false });
    let mime = 'image/png';
    let ext = 'png';

    if (fmt === 'jpg' || fmt === 'jpeg') {
      pipeline = pipeline.flatten({ background: '#ffffff' }).jpeg({ quality: 94, mozjpeg: true });
      mime = 'image/jpeg';
      ext = 'jpg';
    } else if (fmt === 'webp') {
      pipeline = pipeline.webp({ quality: 94 });
      mime = 'image/webp';
      ext = 'webp';
    } else {
      pipeline = pipeline.png({ compressionLevel: 9 });
      mime = 'image/png';
      ext = 'png';
    }

    const out = await pipeline.toBuffer();
    if (!out || !out.length) throw new Error('轉檔輸出為空');

    res.status(200);
    res.setHeader('Content-Type', mime);
    res.setHeader('Content-Length', String(out.length));
    res.setHeader('Content-Disposition', `attachment; filename="urban-upscale-${Date.now()}.${ext}"`);
    res.setHeader('Cache-Control', 'no-store');
    res.end(out);
  } catch (e) {
    console.error('[download error]', e);
    res.status(500).json({ error: '下載轉檔失敗', detail: String(e.message || e) });
  }
});

app.get('*', (_req, res) => {
  res.sendFile(path.join(process.cwd(), 'index.html'));
});

app.listen(PORT, () => console.log('Server running:', PORT));