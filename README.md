# Premium Upscaler 可用版

## 本機啟動
```bash
npm install
cp .env.example .env
npm start
```

打開：
http://localhost:3000

## Render 設定
Build Command:
```bash
npm install
```

Start Command:
```bash
npm start
```

Environment Variables:
```env
REPLICATE_API_TOKEN=你的 token
```

## 說明
這版包含：
- index.html 前端
- server.js 後端
- Natural / Enhance / Text / Illustration pipeline
- 大圖自動縮小防 OOM
- Replicate Topaz image-upscale API 對接
