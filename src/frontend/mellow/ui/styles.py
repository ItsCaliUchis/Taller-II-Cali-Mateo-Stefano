CSS = r"""
@import url('https://fonts.googleapis.com/css2?family=DM+Sans:wght@400;500;600;700&family=Nunito:wght@500;600;700;800;900&display=swap');
:root {
  --ink:#39364d; --muted:#89869d; --lav:#aaa2d8; --lav-dark:#8178b8;
  --pink:#e9b6c6; --mint:#a9d1c4; --paper:#fbfaff; --line:#eeeaf5;
  --shadow:0 12px 34px rgba(77,65,111,.07);
}
.gradio-container {background:linear-gradient(135deg,#fbfaff 0%,#f8f7ff 55%,#fff9fc 100%) !important; color:var(--ink); font-family:'DM Sans',sans-serif !important; max-width:100% !important;}
.gradio-container .main {max-width:1440px !important; padding:24px 28px 52px !important;}
#app-shell {max-width:1280px;margin:0 auto;}
#topbar {display:flex;align-items:center;justify-content:space-between;padding:10px 4px 24px;border-bottom:1px solid var(--line);margin-bottom:26px;}
.brand {display:flex;align-items:center;gap:12px;}
.brand-mark {width:43px;height:43px;border-radius:15px;background:linear-gradient(145deg,#c9c2ed,#f1c9d7);display:grid;place-items:center;color:white;font-size:23px;box-shadow:0 7px 18px #bcb2df55;}
.brand-name {font:900 22px 'Nunito',sans-serif;letter-spacing:-.7px;color:#49445f;}
.brand-sub {font-size:11px;color:#a19cb2;margin-top:1px;letter-spacing:.4px;}
.top-status {background:#fff;border:1px solid var(--line);padding:8px 13px;border-radius:999px;color:#89849c;font-size:12px;display:flex;gap:8px;align-items:center;}
.status-dot {width:7px;height:7px;background:#e7b66f;border-radius:50%;display:inline-block;}
#sidebar {background:rgba(255,255,255,.78);border:1px solid var(--line);border-radius:22px;padding:15px;box-shadow:var(--shadow);min-height:660px;}
.nav-caption,.eyebrow {font-size:10px;font-weight:800;letter-spacing:1.35px;color:#aaa4ba;text-transform:uppercase;}
.nav-caption {padding:10px 12px 9px;}
.nav-btn button {border:0 !important;background:transparent !important;color:#77738c !important;text-align:left !important;border-radius:12px !important;font-weight:600 !important;box-shadow:none !important;padding:12px 14px !important;transition:.18s ease;}
.nav-btn button:hover {background:#f5f2fc !important;color:#6f65a9 !important;transform:translateX(2px);}
.nav-btn.active button {background:#f0edfb !important;color:#7066ad !important;}
.sidebar-note {margin-top:28px;padding:14px;border-radius:15px;background:linear-gradient(145deg,#f8f5ff,#fff5f8);font-size:11px;color:#938da6;line-height:1.6;}
.sidebar-note b {display:block;color:#77709d;margin-bottom:3px;}
#page-content {padding:2px 8px 30px 22px;}
.page-heading {margin-bottom:22px;}
.page-heading h1 {font:900 29px 'Nunito',sans-serif;color:#49445f;letter-spacing:-.8px;margin:0 0 5px;}
.page-heading p {font-size:13px;color:#9691a7;margin:0;}
.hero {border-radius:24px;padding:28px 30px;background:linear-gradient(110deg,#e9e5fa 0%,#f4e8f2 57%,#fceef1 100%);position:relative;overflow:hidden;margin-bottom:22px;min-height:170px;}
.hero:after {content:'✿';position:absolute;right:8%;top:-32px;font-size:180px;color:#fff8;transform:rotate(18deg);}
.hero h2 {font:900 24px 'Nunito',sans-serif;color:#5c557d;margin:0 0 7px;position:relative;z-index:1;}
.hero p {max-width:520px;color:#817a9b;font-size:13px;line-height:1.65;margin:0;position:relative;z-index:1;}
.hero .hero-chip {display:inline-block;margin-top:15px;background:#ffffffa8;border:1px solid #fff9;padding:6px 11px;border-radius:999px;font-size:10px;color:#8176ae;font-weight:700;position:relative;z-index:1;}
.soft-card {background:#fff;border:1px solid var(--line);border-radius:20px;padding:21px;box-shadow:var(--shadow);margin-bottom:16px;}
.card-title {font:800 18px 'Nunito',sans-serif;color:rgb(241, 201, 215);margin:0 0 4px;}
.card-sub {font-size:11px;color:#a09bae;margin-bottom:16px;}
.section-kicker {font-size:10px;font-weight:800;letter-spacing:1px;text-transform:uppercase;color:#aaa4ba;margin-bottom:10px;}
.stat-card {background:#fff;border:1px solid var(--line);border-radius:17px;padding:17px 18px;box-shadow:var(--shadow);}
.stat-value {font:900 24px 'Nunito',sans-serif;color:#625b83;}
.stat-label {font-size:11px;color:#a09bae;margin-top:3px;}
.gradio-container label,.gradio-container .wrap label {color:#77718c ;font-size:12px;font-weight:700 !important;}
.gradio-container textarea,.gradio-container input,.gradio-container select {border:1px solid #eae6f2 !important;border-radius:13px !important;background:#fdfcff !important;color:#514c68 !important;}
.gradio-container textarea:focus,.gradio-container input:focus {border-color:#b9b0df !important;box-shadow:0 0 0 3px #c9c1eb33 !important;}
.gradio-container button.primary {background:linear-gradient(100deg,#9187c8,#b6a6d9) !important;border:0 !important;color:white !important;border-radius:13px !important;font-weight:800 !important;box-shadow:0 7px 16px #9e94cc35 !important;}
.gradio-container button.primary:hover {filter:brightness(1.04);transform:translateY(-1px);}
.gradio-container button.secondary {background:#f5f2fb !important;border:1px solid #ebe6f4 !important;color:#7971a1 !important;border-radius:12px !important;}
.result-card {background:#fff;border:1px solid var(--line);border-radius:19px;padding:20px;box-shadow:var(--shadow);}
.result-topline {display:flex;justify-content:space-between;align-items:center;margin-bottom:18px;}
.demo-tag {font-size:9px;font-weight:800;letter-spacing:.8px;color:#a28cbb;background:#f5effa;padding:5px 8px;border-radius:99px;}
.result-verdict {display:flex;gap:12px;align-items:center;padding:14px;border-radius:14px;background:#f8f6fc;margin-bottom:20px;}
.verdict-dot {width:12px;height:12px;border-radius:50%;background:#aaa;}
.result-verdict.positive .verdict-dot {background:#80b9a8;box-shadow:0 0 0 5px #80b9a822;}
.result-verdict.negative .verdict-dot {background:#e78b9b;box-shadow:0 0 0 5px #e78b9b22;}
.result-verdict.neutral .verdict-dot {background:#a5a0c9;box-shadow:0 0 0 5px #a5a0c922;}
.result-verdict small {display:block;color:#9993a9;font-size:10px;margin-bottom:2px;}
.result-verdict strong {font:900 20px 'Nunito',sans-serif;color:#5a5573;}
.prob-title {font-size:11px;font-weight:800;color:#79738d;margin-bottom:14px;}
.prob-row {margin:11px 0;}
.prob-meta {display:flex;justify-content:space-between;font-size:11px;color:#817c92;margin-bottom:6px;}
.prob-meta b {color:#625c78;}
.prob-track {height:8px;background:#f1eff6;border-radius:20px;overflow:hidden;}
.prob-fill {height:100%;display:block;border-radius:20px;}
.prob-fill.positive {background:#9acdbd;}.prob-fill.neutral {background:#aaa4d0;}.prob-fill.negative {background:#e8a0ae;}
.result-note {font-size:10px;color:#aaa4b5;line-height:1.5;margin:17px 0 0;}
.empty-result,.empty-state {background:#fff;border:1px dashed #e5dfef;border-radius:17px;padding:27px;text-align:center;color:#9992ac;font-size:12px;line-height:1.7;}
.empty-state span {color:#b1abba;font-size:11px;}
.batch-summary {display:grid;grid-template-columns:repeat(4,1fr);gap:9px;margin:8px 0 12px;}
.batch-summary div {background:#faf9fd;border:1px solid #efebf5;border-radius:13px;padding:13px;}
.batch-summary strong {display:block;font:900 22px 'Nunito',sans-serif;color:#625b83;}
.batch-summary span {font-size:10px;color:#a09bae;}
.muted {font-size:10px;color:#a09bae;}
.error-box {background:#fff4f5;border:1px solid #f3d6dc;color:#b66d7c;border-radius:12px;padding:13px;font-size:12px;}
.history-list {display:flex;flex-direction:column;gap:10px;}
.history-item {background:#fff;border:1px solid var(--line);border-radius:14px;padding:14px 16px;}
.history-item>div {display:flex;justify-content:space-between;align-items:center;gap:10px;}
.history-item small {font-size:10px;color:#aaa4b5;}
.history-item p {font-size:11px;color:#77718a;line-height:1.5;margin:9px 0 0;}
.history-label {font-size:10px;font-weight:800;color:#6f9f91;background:#edf7f3;padding:5px 9px;border-radius:99px;}
.history-label.negative {color:#bd7d8a;background:#fff0f2;}
.worker-pill {font-size:10px;padding:5px 9px;border-radius:99px;background:#edf7f3;color:#6f9f91;font-weight:800;}
.worker-pill.off {background:#fff4e8;color:#b28b5b;}
.worker-row {display:flex;justify-content:space-between;align-items:center;gap:15px;}
.worker-name {font:800 13px 'Nunito',sans-serif;color:#625b78;}
.worker-meta {font-size:10px;color:#a09bae;margin-top:4px;}
.footer-note {text-align:center;color:#b1abba;font-size:10px;padding:22px 0 0;}
@media(max-width:850px) {
  .gradio-container .main {padding:14px !important;}
  #sidebar {min-height:unset;margin-bottom:14px;}
  #page-content {padding:8px 0;}
  .hero {padding:22px;}.hero h2 {font-size:21px;}
  .batch-summary {grid-template-columns:repeat(2,1fr);}
}
"""