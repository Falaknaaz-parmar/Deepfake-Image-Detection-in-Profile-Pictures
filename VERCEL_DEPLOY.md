# Deploying to Vercel

This project is configured for deployment to Vercel in Demo Mode.

## Demo Mode

The Vercel deployment runs in **Demo Mode** which:
- ✅ Shows the full web interface
- ✅ Accepts image uploads
- ✅ Returns simulated predictions
- ⚠️ Does not use the actual trained model (to avoid 46MB size limit)

## Quick Deploy

[![Deploy with Vercel](https://vercel.com/button)](https://vercel.com/new/clone?repository-url=https://github.com/Falaknaaz-parmar/Deepfake-Image-Detection-in-Profile-Pictures)

## Manual Deployment

### Prerequisites
- Vercel account ([signup free](https://vercel.com/signup))
- Vercel CLI installed

### Steps

1. **Install Vercel CLI**
```bash
npm install -g vercel
```

2. **Login to Vercel**
```bash
vercel login
```

3. **Deploy**
```bash
vercel --prod
```

4. **Access your app**
Your app will be live at: `https://your-project.vercel.app`

## Configuration

The deployment uses:
- `vercel.json` - Vercel configuration
- `api/index.py` - Serverless function (demo mode)
- `requirements-vercel.txt` - Lightweight dependencies

## Upgrade to Production

To enable real predictions:
1. Host your trained model (`best_model.pth`) externally
2. Update `api/index.py` to download and load the model
3. Set `DEMO_MODE=false` in Vercel environment variables

## Files Structure for Vercel

```
├── api/
│   └── index.py          # Serverless function
├── static/              # CSS, JS files
├── templates/           # HTML templates
├── vercel.json         # Vercel config
└── requirements-vercel.txt  # Dependencies
```
