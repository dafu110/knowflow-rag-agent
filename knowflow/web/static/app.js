let sessionId = Math.random().toString(36).slice(2);
let lastAnswer = null;
let apiToken = '';
const summarySlot = document.getElementById('summarySlot');
const answerTab = document.getElementById('tab-answer');
const citationsTab = document.getElementById('tab-citations');
const traceSlot = document.getElementById('traceSlot');
const modePill = document.getElementById('modePill');
const sessionPill = document.getElementById('sessionPill');
const statsPill = document.getElementById('stats');
const askBtn = document.getElementById('askBtn');
const copyBtn = document.getElementById('copyBtn');
const evalBtn = document.getElementById('evalBtn');
const newSessionBtn = document.getElementById('newSessionBtn');
const refreshDocsBtn = document.getElementById('refreshDocsBtn');
const uploadFormEl = document.getElementById('uploadForm');
const docList = document.getElementById('docList');
const metricDocs = document.getElementById('metricDocs');
const metricChunks = document.getElementById('metricChunks');
const questionInput = document.getElementById('question');
const userInput = document.getElementById('user');
const rolesInput = document.getElementById('roles');
const statusLive = document.getElementById('statusLive');

function escapeHtml(str) {
  return String(str).replace(/[&<>"']/g, s => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[s]));
}

function renderSession() {
  sessionPill.textContent = `session ${sessionId.slice(0, 8)}`;
}

function setMode(value) {
  modePill.textContent = value;
  document.body.dataset.mode = value.replace(/\s+/g, '-').toLowerCase();
  statusLive.textContent = value;
  document.querySelector('.dashboard').setAttribute('aria-busy', String(['retrieving', 'uploading', 'evaluating'].includes(value)));
}

function pct(value) {
  const numeric = Number(value || 0);
  return Math.max(0, Math.min(100, Math.round(numeric * 100)));
}

function scorePct(value) {
  return `${pct(value)}%`;
}

function scoreLevel(value) {
  return Number(value || 0) >= .8 ? 'low' : (Number(value || 0) >= .5 ? 'medium' : 'high');
}

function answerTypeLabel(value) {
  if (value === 'grounded') return '&#x5df2;&#x5f15;&#x7528;';
  if (value === 'clarify') return '&#x9700;&#x6f84;&#x6e05;';
  if (value === 'refusal') return '&#x672a;&#x627e;&#x5230;';
  return escapeHtml(value || '-');
}

function riskLabel(value) {
  if (value === 'low') return '&#x4f4e;';
  if (value === 'medium') return '&#x4e2d;';
  if (value === 'high') return '&#x9ad8;';
  return escapeHtml(value || '-');
}

function renderError(title, message) {
  copyBtn.disabled = true;
  summarySlot.innerHTML = `<div class="meta"><span class="pill" data-level="high">${escapeHtml(title)}</span></div>`;
  answerTab.innerHTML = `<div class="citation"><strong>${escapeHtml(title)}</strong><div>${escapeHtml(message)}</div></div>`;
  citationsTab.innerHTML = '<div class="empty">&#x672c;&#x6b21;&#x64cd;&#x4f5c;&#x672a;&#x4ea7;&#x751f;&#x5f15;&#x7528;&#x3002;</div>';
  traceSlot.innerHTML = '<div class="empty">&#x8bf7;&#x68c0;&#x67e5;&#x672c;&#x5730;&#x670d;&#x52a1;&#x6216;&#x8bc4;&#x6d4b;&#x96c6;&#x8def;&#x5f84;&#x3002;</div>';
  switchTab('answer');
}

async function apiFetch(url, options = {}, retry = true) {
  const headers = new Headers(options.headers || {});
  if (apiToken) headers.set('x-knowflow-token', apiToken);
  const response = await fetch(url, { ...options, headers });
  if (response.status === 401 && retry) {
    const entered = window.prompt('API token');
    if (entered) {
      apiToken = entered;
      return apiFetch(url, options, false);
    }
  }
  return response;
}

async function parseApiResponse(response) {
  const text = await response.text();
  let payload = {};
  if (text) {
    try {
      payload = JSON.parse(text);
    } catch {
      payload = { error: { message: text } };
    }
  }
  if (!response.ok) {
    const message = payload?.error?.message || payload?.message || response.statusText || 'request failed';
    throw new Error(message);
  }
  return payload;
}

function setBusy(isBusy) {
  askBtn.disabled = isBusy;
  evalBtn.disabled = isBusy;
  uploadFormEl.querySelectorAll('input, button').forEach(element => {
    element.disabled = isBusy;
  });
}

function renderWaitingAnswer() {
  answerTab.innerHTML = `
    <div class="trust-grid">
      <div class="trust-row">
        <div class="trust-dot">1</div>
        <div><div class="trust-title">&#x7b54;&#x6848;&#x751f;&#x6210;</div><div class="trust-copy">&#x57fa;&#x4e8e;&#x53ef;&#x89c1;&#x6587;&#x6863;&#x548c;&#x91cd;&#x6392;&#x7ed3;&#x679c;&#x7ec4;&#x7ec7;&#x7b54;&#x6848;&#x3002;</div></div>
        <div class="trust-state">waiting</div>
      </div>
      <div class="trust-row">
        <div class="trust-dot">2</div>
        <div><div class="trust-title">&#x6765;&#x6e90;&#x5f15;&#x7528;</div><div class="trust-copy">&#x4f18;&#x5148;&#x663e;&#x793a;&#x6587;&#x6863;&#x6807;&#x9898;&#x3001;chunk &#x548c;&#x539f;&#x6587;&#x7247;&#x6bb5;&#x3002;</div></div>
        <div class="trust-state">0</div>
      </div>
      <div class="trust-row">
        <div class="trust-dot">3</div>
        <div><div class="trust-title">&#x98ce;&#x9669;&#x68c0;&#x67e5;</div><div class="trust-copy">&#x5f31;&#x8bc1;&#x636e;&#x95ee;&#x9898;&#x4f1a;&#x89e6;&#x53d1;&#x6f84;&#x6e05;&#x6216;&#x62d2;&#x7b54;&#x3002;</div></div>
        <div class="trust-state">guarded</div>
      </div>
    </div>
  `;
}

function renderLoading(message, detail) {
  summarySlot.innerHTML = `
    <div class="loading-card">
      <div class="loading-head">
        <span class="loading-spinner" aria-hidden="true"></span>
        <div>
          <strong>${escapeHtml(message)}</strong>
          <div>${escapeHtml(detail || '正在处理请求')}</div>
        </div>
      </div>
      <div class="loading-bars" aria-hidden="true">
        <span></span><span></span><span></span>
      </div>
    </div>
  `;
  answerTab.innerHTML = `
    <div class="trust-grid">
      <div class="trust-row active">
        <div class="trust-dot">1</div>
        <div><div class="trust-title">权限过滤</div><div class="trust-copy">确认当前用户可见的知识范围。</div></div>
        <div class="trust-state">running</div>
      </div>
      <div class="trust-row active">
        <div class="trust-dot">2</div>
        <div><div class="trust-title">混合检索</div><div class="trust-copy">合并关键词、向量和重排信号。</div></div>
        <div class="trust-state">running</div>
      </div>
      <div class="trust-row">
        <div class="trust-dot">3</div>
        <div><div class="trust-title">引用校验</div><div class="trust-copy">生成回答前检查证据支撑。</div></div>
        <div class="trust-state">queued</div>
      </div>
    </div>
  `;
  switchTab('answer');
}


async function refreshStats() {
  const res = await apiFetch('/health');
  const data = await parseApiResponse(res);
  statsPill.textContent = `${data.stats.documents} docs / ${data.stats.chunks} chunks`;
  metricDocs.textContent = data.stats.documents;
  metricChunks.textContent = data.stats.chunks;
}

async function refreshDocuments() {
  const res = await apiFetch('/documents');
  const data = await parseApiResponse(res);
  if (!data.documents.length) {
    docList.innerHTML = '<div class="empty">&#x6682;&#x65e0;&#x6587;&#x6863;&#x3002;</div>';
    return;
  }
  docList.innerHTML = data.documents.map(doc => `
    <article class="doc-row">
      <div class="doc-head">
        <div class="doc-main">
          <div class="doc-title">${escapeHtml(doc.title)}</div>
          <div class="doc-tags">
            <span class="doc-tag">${doc.chunk_count} chunks</span>
            <span class="doc-tag">roles ${(doc.allowed_roles || []).length ? escapeHtml(doc.allowed_roles.join(', ')) : 'public'}</span>
          </div>
        </div>
      </div>
      <details class="doc-details">
        <summary class="advanced-summary">&#x67e5;&#x770b;&#x8be6;&#x60c5;</summary>
        <div class="doc-meta">${escapeHtml(doc.source)}</div>
        <div class="doc-meta">users ${escapeHtml((doc.allowed_users || []).join(', ') || 'any')}</div>
        <div class="doc-actions"><button class="danger" type="button" data-delete-doc="${escapeHtml(doc.id)}">&#x5220;&#x9664;</button></div>
      </details>
    </article>
  `).join('');
}

function switchTab(name) {
  document.querySelectorAll('.tab').forEach(tab => {
    const active = tab.dataset.tab === name;
    tab.classList.toggle('active', active);
    tab.setAttribute('aria-selected', String(active));
    tab.tabIndex = active ? 0 : -1;
  });
  document.querySelectorAll('.tab-panel').forEach(panel => {
    const active = panel.id === `tab-${name}`;
    panel.classList.toggle('active', active);
    panel.hidden = !active;
  });
}

function renderUploadSuccess(data) {
  copyBtn.disabled = true;
  summarySlot.innerHTML = `
    <div class="meta">
      <span class="pill" data-level="low">&#x4e0a;&#x4f20;&#x6210;&#x529f;</span>
      <span class="pill">${data.added} chunks indexed</span>
      <span class="pill">${data.stats.documents} docs / ${data.stats.chunks} chunks</span>
    </div>
  `;
  answerTab.innerHTML = `
    <div class="citation">
      <strong>&#x6587;&#x6863;&#x5df2;&#x52a0;&#x5165;&#x77e5;&#x8bc6;&#x5e93;</strong>
      <div>&#x7cfb;&#x7edf;&#x5df2;&#x5b8c;&#x6210;&#x5207;&#x5206;&#x548c;&#x7d22;&#x5f15;&#xff0c;&#x53ef;&#x4ee5;&#x76f4;&#x63a5;&#x7528;&#x53f3;&#x4fa7;&#x95ee;&#x7b54;&#x9a8c;&#x8bc1;&#x68c0;&#x7d22;&#x6548;&#x679c;&#x3002;</div>
    </div>
  `;
  citationsTab.innerHTML = '<div class="empty">&#x4e0a;&#x4f20;&#x64cd;&#x4f5c;&#x4e0d;&#x4ea7;&#x751f;&#x56de;&#x7b54;&#x5f15;&#x7528;&#xff0c;&#x8bf7;&#x63d0;&#x95ee;&#x540e;&#x67e5;&#x770b;&#x6765;&#x6e90;&#x3002;</div>';
  traceSlot.innerHTML = '<div class="empty">&#x65b0;&#x6587;&#x6863;&#x5df2;&#x8fdb;&#x5165;&#x6df7;&#x5408;&#x68c0;&#x7d22;&#x5019;&#x9009;&#x96c6;&#x3002;</div>';
  switchTab('answer');
}

function renderResult(data) {
  copyBtn.disabled = false;
  const strongCount = (data.retrieval_debug || []).filter(item => item.evidence_grade === 'strong').length;
  const citationCount = (data.citations || []).length;
  const retrievalCount = (data.retrieval_debug || []).length;
  summarySlot.innerHTML = `
    <div class="result-summary">
      <div class="result-kicker">
        <span>&#x672c;&#x6b21;&#x56de;&#x7b54;&#x7684;&#x4f9d;&#x636e;&#x6982;&#x89c8;</span>
        <span>${citationCount} citations</span>
      </div>
      <div class="answer-metrics">
        <div class="answer-metric" data-level="${data.answer_type === 'grounded' ? 'low' : (data.answer_type === 'clarify' ? 'medium' : 'high')}">
          <div class="answer-metric-label">&#x56de;&#x7b54;&#x72b6;&#x6001;</div>
          <div class="answer-metric-value">${answerTypeLabel(data.answer_type)}</div>
        </div>
        <div class="answer-metric" data-level="${scoreLevel(data.confidence)}">
          <div class="answer-metric-label">&#x7f6e;&#x4fe1;&#x5ea6;</div>
          <div class="answer-metric-value">${scorePct(data.confidence)}</div>
        </div>
        <div class="answer-metric" data-level="${data.hallucination_risk}">
          <div class="answer-metric-label">&#x5e7b;&#x89c9;&#x98ce;&#x9669;</div>
          <div class="answer-metric-value">${riskLabel(data.hallucination_risk)}</div>
        </div>
        <div class="answer-metric" data-level="${strongCount ? 'low' : 'medium'}">
          <div class="answer-metric-label">&#x5f3a;&#x8bc1;&#x636e;</div>
          <div class="answer-metric-value">${strongCount}/${retrievalCount}</div>
        </div>
      </div>
    </div>
  `;

  const followUps = (data.follow_up_questions || []).length
    ? `<div class="citation"><strong>&#x5efa;&#x8bae;&#x8ffd;&#x95ee;</strong><div class="follow-up-actions">${data.follow_up_questions.map(q => `<button class="chip follow-up-chip" type="button" data-q="${escapeHtml(q)}">${escapeHtml(q)}</button>`).join('')}</div></div>`
    : '';
  const evidence = data.evidence_summary
    ? `<div class="citation"><strong>&#x8bc1;&#x636e;&#x6458;&#x8981;</strong><div>${escapeHtml(data.evidence_summary)}</div></div>`
    : '';
  const refusal = data.answer_type === 'refusal'
    ? `<div class="guardrail-notice"><strong>权限或证据边界已生效</strong><div>当前身份下没有可引用的可靠依据，因此系统没有扩展回答，也没有暴露受限文档。</div></div>`
    : '';
  const inlineEvidence = citationCount
    ? `<section class="evidence-list" aria-label="回答依据原文"><strong>回答依据原文</strong>${data.citations.slice(0, 2).map(c => `
      <blockquote class="evidence-quote"><span>${escapeHtml(c.title)} · ${escapeHtml(c.chunk_id)}</span><p>${escapeHtml(c.quote)}</p></blockquote>`).join('')}</section>`
    : '';
  answerTab.innerHTML = `${refusal}<div class="answer">${escapeHtml(data.answer)}</div>${inlineEvidence}${evidence}${followUps}`;

  citationsTab.innerHTML = (data.citations || []).length
    ? data.citations.map(c => `
      <div class="citation">
        <strong>${escapeHtml(c.title)}</strong>
        <div>${escapeHtml(c.chunk_id)}</div>
        <p>${escapeHtml(c.quote)}</p>
      </div>`).join('')
    : '<div class="empty">&#x6ca1;&#x6709;&#x53ef;&#x7528;&#x5f15;&#x7528;&#x3002;</div>';

  traceSlot.innerHTML = (data.retrieval_debug || []).length
    ? data.retrieval_debug.map(item => `
      <div class="trace-row ${item.evidence_grade === 'strong' ? 'strong' : ''}">
        <div class="trace-head">
          <span>${escapeHtml(item.evidence_grade === 'strong' ? '强证据' : '弱证据')}</span>
          <span>score ${item.score}</span>
        </div>
        <div class="trace-meta">${escapeHtml(item.source)} - ${escapeHtml(item.chunk_id)}</div>
        <div class="trace-meta">&#x547d;&#x4e2d;&#x539f;&#x56e0;&#xff1a;${escapeHtml((item.reasons || []).join(', '))}</div>
        <details>
          <summary class="advanced-summary">&#x67e5;&#x770b;&#x6280;&#x672f;&#x5206;&#x6570;</summary>
          <div class="score-bars">
            <div class="score-line"><span>BM25</span><span class="score-track"><span class="score-fill" style="width:${pct(item.bm25)}%"></span></span><span>${pct(item.bm25)}%</span></div>
            <div class="score-line"><span>Vector</span><span class="score-track"><span class="score-fill vector" style="width:${pct(item.vector)}%"></span></span><span>${pct(item.vector)}%</span></div>
            <div class="score-line"><span>Rerank</span><span class="score-track"><span class="score-fill rerank" style="width:${pct(item.rerank)}%"></span></span><span>${pct(item.rerank)}%</span></div>
          </div>
        </details>
      </div>`).join('')
    : '<div class="empty">&#x8fd8;&#x6ca1;&#x6709;&#x68c0;&#x7d22;&#x8bb0;&#x5f55;&#x3002;</div>';
  switchTab('answer');
}

function renderEval(data) {
  copyBtn.disabled = true;
  const leakLevel = data.permission_leaks > 0 ? 'high' : 'low';
  summarySlot.innerHTML = `
    <div class="meta">
      <span class="pill" data-level="low">&#x79bb;&#x7ebf;&#x8bc4;&#x6d4b;&#x5b8c;&#x6210;</span>
      <span class="pill">${data.total} cases</span>
      <span class="pill" data-level="${leakLevel}">${data.permission_leaks} leaks</span>
    </div>
  `;
  answerTab.innerHTML = `
    <div class="eval-grid">
      <div class="eval-card"><div class="eval-label">Recall@K</div><div class="eval-value">${scorePct(data.recall_at_k)}</div><div class="eval-note">&#x9884;&#x671f;&#x6765;&#x6e90;&#x662f;&#x5426;&#x88ab;&#x627e;&#x5230;</div></div>
      <div class="eval-card"><div class="eval-label">MRR</div><div class="eval-value">${scorePct(data.mrr)}</div><div class="eval-note">&#x6b63;&#x786e;&#x6765;&#x6e90;&#x6392;&#x540d;&#x8d8a;&#x524d;&#x8d8a;&#x597d;</div></div>
      <div class="eval-card"><div class="eval-label">&#x5f15;&#x7528;&#x51c6;&#x786e;&#x7387;</div><div class="eval-value">${scorePct(data.citation_accuracy)}</div><div class="eval-note">&#x56de;&#x7b54;&#x662f;&#x5426;&#x5f15;&#x5230;&#x6b63;&#x786e;&#x6587;&#x6863;</div></div>
      <div class="eval-card"><div class="eval-label">&#x5fe0;&#x5b9e;&#x5ea6;</div><div class="eval-value">${scorePct(data.faithfulness)}</div><div class="eval-note">&#x662f;&#x5426;&#x51fa;&#x73b0;&#x672a;&#x652f;&#x6491;&#x65ad;&#x8a00;</div></div>
      <div class="eval-card"><div class="eval-label">&#x6743;&#x9650;&#x6cc4;&#x6f0f;</div><div class="eval-value">${data.permission_leaks}</div><div class="eval-note">&#x68c0;&#x7d22;&#x6216;&#x5f15;&#x7528;&#x547d;&#x4e2d;&#x7981;&#x6b62;&#x6765;&#x6e90;</div></div>
    </div>
    <div class="citation">
      <strong>&#x8bc4;&#x6d4b;&#x89e3;&#x8bfb;</strong>
      <div>&#x8fd9;&#x4e2a;&#x9762;&#x677f;&#x7528;&#x540c;&#x4e00;&#x5957; Agent &#x94fe;&#x8def;&#x56de;&#x653e;&#x79bb;&#x7ebf;&#x95ee;&#x9898;&#xff0c;&#x540c;&#x65f6;&#x68c0;&#x67e5;&#x68c0;&#x7d22;&#x8d28;&#x91cf;&#x3001;&#x5f15;&#x7528;&#x548c;&#x6743;&#x9650;&#x8fb9;&#x754c;&#x3002;</div>
    </div>
    ${Object.keys(data.scenario_summary || {}).length ? `<div class="scenario-list">${Object.entries(data.scenario_summary).map(([name, item]) => `<span class="pill" data-level="${item.permission_leaks ? 'high' : 'low'}">${escapeHtml(name)} ${item.passed}/${item.total}</span>`).join('')}</div>` : ''}
  `;
  citationsTab.innerHTML = (data.cases || []).map((item, index) => `
    <div class="case-row">
      <div class="case-title">${index + 1}. ${escapeHtml(item.question)}</div>
      <div class="case-meta">
        <span class="pill" data-level="${item.recall_hit ? 'low' : 'high'}">recall ${item.recall_hit ? 'hit' : 'miss'}</span>
        <span class="pill" data-level="${item.permission_leak ? 'high' : 'low'}">leak ${item.permission_leak ? 'yes' : 'no'}</span>
        <span class="pill" data-level="${scoreLevel(item.term_coverage)}">terms ${scorePct(item.term_coverage)}</span>
        <span class="pill">rank ${item.first_rank || '-'}</span>
      </div>
      <div class="trace-meta">&#x68c0;&#x7d22;&#xff1a;${escapeHtml((item.retrieved_sources || []).join(', ') || '-')}</div>
      <div class="trace-meta">&#x5f15;&#x7528;&#xff1a;${escapeHtml((item.cited_sources || []).join(', ') || '-')}</div>
    </div>
  `).join('');
  traceSlot.innerHTML = `
    <div class="status-card">
      <div class="step-dot">E</div>
      <div><div class="step-title">&#x79bb;&#x7ebf;&#x8bc4;&#x6d4b;&#x96c6;</div><div class="step-copy">&#x6309;&#x56fa;&#x5b9a;&#x95ee;&#x9898;&#x96c6;&#x91cd;&#x653e;&#x68c0;&#x7d22;&#x3001;&#x91cd;&#x6392;&#x3001;&#x5f15;&#x7528;&#x548c;&#x6743;&#x9650;&#x68c0;&#x67e5;&#x3002;</div></div>
    </div>
  `;
  switchTab('answer');
}

uploadFormEl.addEventListener('submit', async (event) => {
  event.preventDefault();
  const form = new FormData(uploadFormEl);
  setMode('uploading');
  setBusy(true);
  renderLoading('正在上传并切分文档', '解析元数据、生成 chunks 并刷新知识库');
  try {
    const res = await apiFetch('/upload', { method: 'POST', body: form });
    const data = await parseApiResponse(res);
    renderUploadSuccess(data);
    setMode('ready');
    await refreshStats();
    await refreshDocuments();
  } catch (error) {
    renderError('upload failed', error.message || String(error));
    setMode('error');
  } finally {
    setBusy(false);
  }
});

askBtn.addEventListener('click', async () => {
  if (!questionInput.value.trim()) {
    renderError('ask failed', '问题不能为空。');
    setMode('error');
    return;
  }
  setMode('retrieving');
  setBusy(true);
  renderLoading('Agent 正在检索证据', '先做权限过滤，再召回、重排和引用校验');
  try {
    const res = await apiFetch('/ask', {
      method: 'POST',
      headers: { 'content-type': 'application/json' },
      body: JSON.stringify({
        question: questionInput.value,
        user: userInput.value,
        roles: rolesInput.value.split(',').map(x => x.trim()).filter(Boolean),
        session_id: sessionId
      })
    });
    const data = await parseApiResponse(res);
    lastAnswer = data;
    renderResult(data);
    setMode(data.answer_type || 'ready');
  } catch (error) {
    renderError('ask failed', error.message || String(error));
    setMode('error');
  } finally {
    setBusy(false);
  }
});

evalBtn.addEventListener('click', async () => {
  setMode('evaluating');
  setBusy(true);
  renderLoading('正在运行离线评测', '回放评测集并计算召回、引用、忠实度和权限泄漏');
  try {
    const res = await apiFetch('/eval', { method: 'POST' });
    const data = await parseApiResponse(res);
    renderEval(data);
    setMode('eval ready');
  } catch (error) {
    renderError('eval failed', error.message || String(error));
    setMode('error');
  } finally {
    setBusy(false);
  }
});

copyBtn.addEventListener('click', async () => {
  if (!lastAnswer) return;
  await navigator.clipboard.writeText(JSON.stringify(lastAnswer, null, 2));
  setMode('copied');
  setTimeout(() => setMode(lastAnswer.answer_type || 'ready'), 900);
});

newSessionBtn.addEventListener('click', () => {
  sessionId = Math.random().toString(36).slice(2);
  lastAnswer = null;
  copyBtn.disabled = true;
  renderSession();
  summarySlot.innerHTML = '<div class="empty">&#x65b0;&#x4f1a;&#x8bdd;&#x5df2;&#x5c31;&#x7eea;&#xff0c;&#x53ef;&#x4ee5;&#x5f00;&#x59cb;&#x65b0;&#x7684;&#x8ffd;&#x95ee;&#x94fe;&#x3002;</div>';
  renderWaitingAnswer();
  citationsTab.innerHTML = '<div class="empty">&#x6682;&#x65e0;&#x5f15;&#x7528;&#x3002;</div>';
  traceSlot.innerHTML = '<div class="empty">&#x8fd8;&#x6ca1;&#x6709;&#x68c0;&#x7d22;&#x8bb0;&#x5f55;&#x3002;</div>';
  setMode('ready');
});

document.addEventListener('click', async (event) => {
  const questionButton = event.target.closest('[data-q]');
  if (questionButton) {
    questionInput.value = questionButton.getAttribute('data-q') || '';
    questionInput.focus();
    return;
  }
  const focusTarget = event.target.closest('[data-focus]');
  if (focusTarget) {
    const target = document.querySelector(focusTarget.getAttribute('data-focus'));
    if (target) target.scrollIntoView({ behavior: 'smooth', block: 'center' });
    return;
  }
  const tab = event.target.closest('[data-tab]');
  if (tab) {
    switchTab(tab.dataset.tab);
    return;
  }
  const deleteButton = event.target.closest('[data-delete-doc]');
  if (deleteButton) {
    const documentId = deleteButton.getAttribute('data-delete-doc');
    if (!confirm('\\u5220\\u9664\\u8fd9\\u4efd\\u6587\\u6863\\u53ca\\u5176\\u6240\\u6709 chunks\\uff1f')) return;
    deleteButton.disabled = true;
    try {
      const response = await apiFetch(`/documents?id=${encodeURIComponent(documentId)}`, { method: 'DELETE' });
      await parseApiResponse(response);
      await refreshStats();
      await refreshDocuments();
      setMode('ready');
    } catch (error) {
      renderError('delete failed', error.message || String(error));
      setMode('error');
    }
  }
});

document.addEventListener('keydown', event => {
  if ((event.ctrlKey || event.metaKey) && event.key === 'Enter' && !askBtn.disabled) {
    askBtn.click();
  }
});

refreshDocsBtn.addEventListener('click', refreshDocuments);
renderSession();
setMode('ready');
refreshStats().catch(error => renderError('health failed', error.message || String(error)));
refreshDocuments().catch(error => renderError('documents failed', error.message || String(error)));
