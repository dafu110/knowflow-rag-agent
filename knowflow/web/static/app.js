let sessionId = Math.random().toString(36).slice(2);
let lastAnswer = null;
let apiToken = window.sessionStorage.getItem('knowflow-token') || '';
let serverIdentityEnabled = false;
let sessionTurns = [];
let sessionOwner = '';
let managementEnabled = true;

const summarySlot = document.getElementById('summarySlot');
const answerTab = document.getElementById('tab-answer');
const citationsTab = document.getElementById('tab-citations');
const traceSlot = document.getElementById('traceSlot');
const inspectorSlot = document.getElementById('inspectorSlot');
const evaluationSlot = document.getElementById('evaluationSlot');
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
const identitySummary = document.getElementById('identitySummary');
const identityDetails = document.getElementById('identityDetails');
const recentQuestionList = document.getElementById('recentQuestionList');
const conversationHistory = document.getElementById('conversationHistory');
const knowledgeNotice = document.getElementById('knowledgeNotice');
const historyLabel = document.getElementById('historyLabel');
const historyHint = document.getElementById('historyHint');
const clearHistoryBtn = document.getElementById('clearHistoryBtn');
const currentSessionName = document.getElementById('currentSessionName');
const currentSessionMeta = document.getElementById('currentSessionMeta');
const shareSessionBtn = document.getElementById('shareSessionBtn');
const navigationDrawer = document.getElementById('navigationDrawer');
const contextDrawer = document.getElementById('contextDrawer');
const drawerBackdrop = document.querySelector('.drawer-backdrop');
const localSessionsKey = 'knowflow-local-sessions';
const localIdentityKey = 'knowflow-local-identity';

function escapeHtml(value) {
  return String(value ?? '').replace(/[&<>"']/g, char => ({ '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' })[char]);
}

function pct(value) {
  return Math.max(0, Math.min(100, Math.round(Number(value || 0) * 100)));
}

function scoreLevel(value) {
  return Number(value || 0) >= .8 ? 'low' : (Number(value || 0) >= .5 ? 'medium' : 'high');
}

function answerTypeLabel(value) {
  return { grounded: '已引用', clarify: '需澄清', refusal: '未找到依据' }[value] || escapeHtml(value || '-');
}

function riskLabel(value) {
  return { low: '低', medium: '中', high: '高' }[value] || escapeHtml(value || '-');
}

function renderSession() {
  sessionPill.textContent = `会话 ${sessionId.slice(0, 8)}`;
  const current = serverIdentityEnabled ? sessionTurns : currentLocalSession()?.turns;
  const title = current?.[0]?.question || '知识库问答';
  currentSessionName.textContent = title.slice(0, 28);
  currentSessionMeta.textContent = serverIdentityEnabled
    ? (sessionOwner && sessionOwner !== userInput.value ? '共享只读' : '团队会话')
    : '本机会话';
}

function setManagementEnabled(enabled) {
  managementEnabled = enabled;
  document.querySelectorAll('[data-view="knowledge"], [data-view="evaluation"], [data-management]').forEach(element => { element.hidden = !enabled; });
}

function updateIdentitySummary() {
  const user = userInput.value.trim() || '当前用户';
  const roles = rolesInput.value.trim() || '默认权限';
  identitySummary.textContent = `${user} · ${roles}`;
}

function resizeQuestionInput() {
  const minimum = window.matchMedia('(max-width: 480px)').matches ? 104 : 112;
  if (!questionInput.value.trim()) {
    questionInput.style.height = `${minimum}px`;
    return;
  }
  questionInput.style.height = 'auto';
  questionInput.style.height = `${Math.min(Math.max(questionInput.scrollHeight, minimum), 180)}px`;
}

function localSessions() {
  try {
    const value = JSON.parse(window.localStorage.getItem(localSessionsKey) || '[]');
    return Array.isArray(value) ? value.filter(item => item && typeof item.id === 'string' && Array.isArray(item.turns)) : [];
  } catch { return []; }
}

function saveLocalSessions(sessions) {
  try { window.localStorage.setItem(localSessionsKey, JSON.stringify(sessions.slice(0, 8))); } catch { return; }
}

function formatLocalTime(value) {
  try { return new Intl.DateTimeFormat('zh-CN', { month: 'numeric', day: 'numeric', hour: '2-digit', minute: '2-digit' }).format(new Date(value)); } catch { return ''; }
}

function renderLocalSessions() {
  const entries = localSessions();
  clearHistoryBtn.hidden = entries.length === 0;
  recentQuestionList.innerHTML = entries.length
    ? entries.map(entry => `<button class="recent-question" type="button" data-local-session="${escapeHtml(entry.id)}"><strong>${escapeHtml(entry.title || '未命名会话')}</strong><span>${entry.turns.length} 轮 · ${escapeHtml(formatLocalTime(entry.updatedAt))}</span></button>`).join('')
    : '<div class="empty">还没有本机记录。</div>';
}

function currentLocalSession() {
  return localSessions().find(entry => entry.id === sessionId) || null;
}

async function saveCurrentTurn(question, answer) {
  if (serverIdentityEnabled) {
    sessionTurns = [...sessionTurns, { question, answer }].slice(-12);
    if (sessionOwner && sessionOwner !== userInput.value) {
      sessionId = Math.random().toString(36).slice(2);
      sessionOwner = userInput.value;
    }
    await saveServerSession();
    renderSession();
    return;
  }
  const sessions = localSessions().filter(entry => entry.id !== sessionId);
  const existing = currentLocalSession();
  const turns = [...(existing?.turns || []), { question, answer }].slice(-12);
  sessions.unshift({ id: sessionId, title: existing?.title || question.slice(0, 32), user: userInput.value.trim(), roles: rolesInput.value.trim(), updatedAt: new Date().toISOString(), turns });
  saveLocalSessions(sessions);
  renderLocalSessions();
  renderSession();
}

function renderConversationHistory(session) {
  const previousTurns = (session?.turns || []).slice(0, -1);
  conversationHistory.hidden = previousTurns.length === 0;
  conversationHistory.innerHTML = previousTurns.map((turn, index) => `<article class="history-turn"><p class="history-question">${index + 1}. ${escapeHtml(turn.question)}</p><p class="history-answer">${escapeHtml(turn.answer?.answer || '没有可恢复的回答。')}</p></article>`).join('');
}

function restoreLocalSession(id) {
  const session = localSessions().find(entry => entry.id === id);
  if (!session || !session.turns.length) return;
  sessionId = session.id;
  sessionTurns = session.turns;
  userInput.value = session.user || userInput.value;
  rolesInput.value = session.roles || rolesInput.value;
  const latestTurn = session.turns[session.turns.length - 1];
  questionInput.value = latestTurn.question || '';
  resizeQuestionInput();
  lastAnswer = latestTurn.answer || null;
  renderSession(); updateIdentitySummary(); renderConversationHistory(session); showView('workspace');
  if (lastAnswer) { renderResult(lastAnswer); setMode(lastAnswer.answer_type || 'ready'); }
}

function loadLocalIdentity() {
  try {
    const saved = JSON.parse(window.localStorage.getItem(localIdentityKey) || '{}');
    if (typeof saved.user === 'string' && saved.user) userInput.value = saved.user;
    if (typeof saved.roles === 'string' && saved.roles) rolesInput.value = saved.roles;
  } catch { return; }
}

function saveLocalIdentity() {
  try { window.localStorage.setItem(localIdentityKey, JSON.stringify({ user: userInput.value.trim(), roles: rolesInput.value.trim() })); } catch { return; }
}

async function loadServerIdentity() {
  try {
    const data = await parseApiResponse(await apiFetch('/identity'));
    if (!data.authenticated) { setManagementEnabled(true); return false; }
    serverIdentityEnabled = true;
    userInput.value = data.user;
    rolesInput.value = (data.roles || []).join(', ');
    sessionOwner = data.user;
    userInput.disabled = true;
    rolesInput.disabled = true;
    identityDetails.hidden = true;
    updateIdentitySummary();
    historyLabel.textContent = '团队会话';
    historyHint.textContent = '与协作者共享的服务端记录';
    clearHistoryBtn.hidden = true;
    shareSessionBtn.hidden = false;
    setManagementEnabled((data.roles || []).includes('admin'));
    renderSession();
    await loadServerSessions();
    return true;
  } catch (error) {
    setManagementEnabled(false);
    if (apiToken) renderError('身份连接失败', error.message || String(error));
    return false;
  }
}

async function loadServerSessions() {
  const data = await parseApiResponse(await apiFetch('/sessions'));
  recentQuestionList.innerHTML = (data.sessions || []).length
    ? data.sessions.map(session => `<button class="recent-question" type="button" data-server-session="${escapeHtml(session.id)}"><strong>${escapeHtml(session.title)}</strong><span>${session.turn_count} 轮 · ${escapeHtml(formatLocalTime(session.updated_at))}${session.owner !== userInput.value ? ' · 共享给我' : ''}</span></button>`).join('')
    : '<div class="empty">还没有团队会话。</div>';
}

async function saveServerSession() {
  const title = (sessionTurns[0]?.question || '未命名会话').slice(0, 32);
  const data = await parseApiResponse(await apiFetch('/sessions', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ id: sessionId, title, turns: sessionTurns }) }));
  sessionOwner = data.session.owner;
  await loadServerSessions();
}

async function restoreServerSession(id) {
  const data = await parseApiResponse(await apiFetch(`/sessions/${encodeURIComponent(id)}`));
  const session = data.session;
  if (!session?.turns?.length) return;
  sessionId = session.id;
  sessionOwner = session.owner || userInput.value;
  sessionTurns = session.turns;
  const latestTurn = sessionTurns[sessionTurns.length - 1];
  questionInput.value = latestTurn.question || '';
  resizeQuestionInput();
  lastAnswer = latestTurn.answer || null;
  shareSessionBtn.hidden = sessionOwner !== userInput.value;
  renderSession(); renderConversationHistory({ turns: sessionTurns }); showView('workspace');
  if (lastAnswer) { renderResult(lastAnswer); setMode(lastAnswer.answer_type || 'ready'); }
}

function setMode(value) {
  const labels = { ready: '就绪', retrieving: '正在检索', uploading: '正在导入', evaluating: '正在评测', error: '需要处理', copied: '已复制', grounded: '已引用', clarify: '需澄清', refusal: '未找到依据', 'eval ready': '评测完成' };
  modePill.textContent = labels[value] || value;
  document.body.dataset.mode = value.replace(/\s+/g, '-').toLowerCase();
  statusLive.textContent = labels[value] || value;
  document.querySelector('.app-content').setAttribute('aria-busy', String(['retrieving', 'uploading', 'evaluating'].includes(value)));
}

function setBusy(isBusy) {
  askBtn.disabled = isBusy;
  evalBtn.disabled = isBusy;
  uploadFormEl.querySelectorAll('input, button').forEach(element => { element.disabled = isBusy; });
}

function showView(name) {
  if (!managementEnabled && ['knowledge', 'evaluation'].includes(name)) return;
  document.querySelectorAll('.view').forEach(view => {
    const active = view.id === `view-${name}`;
    view.classList.toggle('active', active);
    view.hidden = !active;
  });
  document.querySelectorAll('[data-view]').forEach(button => button.classList.toggle('active', button.dataset.view === name));
  closeDrawers();
  if (name === 'workspace') questionInput.focus({ preventScroll: true });
}

function openDrawer(name) {
  const drawer = name === 'context' ? contextDrawer : navigationDrawer;
  drawer.classList.add('open');
  drawer.setAttribute('aria-hidden', 'false');
  drawerBackdrop.hidden = false;
  document.querySelectorAll(`[data-drawer="${name}"]`).forEach(button => button.setAttribute('aria-expanded', 'true'));
}

function closeDrawers() {
  [navigationDrawer, contextDrawer].forEach(drawer => {
    drawer.classList.remove('open');
    drawer.setAttribute('aria-hidden', 'true');
  });
  drawerBackdrop.hidden = true;
  document.querySelectorAll('[data-drawer]').forEach(button => button.setAttribute('aria-expanded', 'false'));
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

async function apiFetch(url, options = {}, retry = true) {
  const headers = new Headers(options.headers || {});
  if (apiToken) headers.set('x-knowflow-token', apiToken);
  const response = await fetch(url, { ...options, headers });
  if (response.status === 401 && retry) {
    const entered = window.prompt('请输入 API token');
    if (entered) {
      apiToken = entered;
      window.sessionStorage.setItem('knowflow-token', entered);
      return apiFetch(url, options, false);
    }
  }
  return response;
}

async function parseApiResponse(response) {
  const text = await response.text();
  let payload = {};
  if (text) {
    try { payload = JSON.parse(text); } catch { payload = { error: { message: text } }; }
  }
  if (!response.ok) throw new Error(payload?.error?.message || payload?.message || response.statusText || 'request failed');
  return payload;
}

async function refreshStats() {
  const data = await parseApiResponse(await apiFetch('/health'));
  statsPill.textContent = `${data.stats.documents} 份文档 / ${data.stats.chunks} 个片段`;
  metricDocs.textContent = data.stats.documents;
  metricChunks.textContent = data.stats.chunks;
}

async function refreshDocuments() {
  const data = await parseApiResponse(await apiFetch('/documents'));
  if (!data.documents.length) {
    docList.innerHTML = '<div class="empty">资料库中还没有文档。</div>';
    return;
  }
  docList.innerHTML = data.documents.map(doc => `
    <article class="doc-row">
      <div class="doc-title">${escapeHtml(doc.title)}</div>
      <div class="doc-tags"><span class="doc-tag">${doc.chunk_count} 个片段</span><span class="doc-tag">${(doc.allowed_roles || []).length ? escapeHtml(doc.allowed_roles.join(', ')) : '公开角色'}</span></div>
      <details class="doc-details"><summary class="advanced-summary">查看访问范围</summary><div class="doc-meta">来源：${escapeHtml(doc.source)}</div><div class="doc-meta">指定用户：${escapeHtml((doc.allowed_users || []).join(', ') || '所有用户')}</div><div class="doc-actions"><button class="danger" type="button" data-delete-doc="${escapeHtml(doc.id)}">删除文档</button></div></details>
    </article>`).join('');
}

function renderInspector(data) {
  const debug = data.retrieval_debug || [];
  const citations = data.citations || [];
  const strongCount = debug.filter(item => item.evidence_grade === 'strong').length;
  const aclLevel = data.answer_type === 'refusal' ? 'danger' : (data.answer_type === 'clarify' ? 'warning' : 'ok');
  const aclText = data.answer_type === 'refusal'
    ? '当前身份下没有足以支撑回答的可见证据。'
    : `访问控制已应用，检索了当前用户和角色可访问的 ${debug.length} 个候选片段。`;
  const sources = (citations.length ? citations : debug).slice(0, 4);
  inspectorSlot.innerHTML = `
    <div class="acl-banner" data-level="${aclLevel}"><strong>访问边界</strong><br>${escapeHtml(aclText)}</div>
    <section class="inspector-section"><h3>本次证据</h3><div class="inspector-summary"><div class="inspector-stat"><span>命中片段</span><strong>${debug.length}</strong></div><div class="inspector-stat"><span>强证据</span><strong>${strongCount}</strong></div><div class="inspector-stat"><span>引用来源</span><strong>${citations.length}</strong></div><div class="inspector-stat"><span>风险</span><strong>${riskLabel(data.hallucination_risk)}</strong></div></div></section>
    <section class="inspector-section"><h3>优先来源</h3>${sources.length ? sources.map(source => `<div class="evidence-source"><strong>${escapeHtml(source.title || source.source || '未命名来源')}</strong><span>${escapeHtml(source.chunk_id || '')}${source.score !== undefined ? ` · 得分 ${escapeHtml(source.score)}` : ''}</span></div>`).join('') : '<div class="empty">没有可展示的来源。</div>'}</section>
    <section class="inspector-section"><h3>回答状态</h3><div class="status-chip" data-level="${data.answer_type === 'grounded' ? 'low' : (data.answer_type === 'clarify' ? 'medium' : 'high')}">${answerTypeLabel(data.answer_type)} · 置信度 ${pct(data.confidence)}%</div></section>`;
}

function renderLoading(message, detail) {
  document.body.classList.add('has-answer');
  document.querySelector('.response-area').classList.add('has-result');
  contextDrawer.hidden = false;
  document.querySelector('.context-toggle').hidden = false;
  summarySlot.innerHTML = `<div class="loading-card"><div class="loading-head"><span class="loading-spinner" aria-hidden="true"></span><div><strong>${escapeHtml(message)}</strong><div>${escapeHtml(detail)}</div></div></div><div class="loading-bars" aria-hidden="true"><span></span><span></span><span></span></div></div>`;
  answerTab.innerHTML = '<div class="empty">正在确认权限、召回资料并校验引用。</div>';
  inspectorSlot.innerHTML = '<div class="inspector-empty"><strong>正在构建证据链</strong><p>先按身份过滤资料，再合并检索与重排信号。</p></div>';
  switchTab('answer');
}

function renderError(title, message) {
  copyBtn.disabled = true;
  document.querySelector('.response-area').classList.add('has-result');
  summarySlot.innerHTML = `<div class="guardrail-notice"><strong>${escapeHtml(title)}</strong><div>${escapeHtml(message)}</div></div>`;
  answerTab.innerHTML = '<div class="empty">本次操作没有生成回答。</div>';
  citationsTab.innerHTML = '<div class="empty">本次操作没有产生引用。</div>';
  traceSlot.innerHTML = '<div class="empty">请检查服务状态或输入内容。</div>';
  inspectorSlot.innerHTML = '<div class="inspector-empty"><strong>未生成证据</strong><p>当前请求没有可展示的检索上下文。</p></div>';
  switchTab('answer');
}

function renderResult(data) {
  document.body.classList.add('has-answer');
  document.querySelector('.response-area').classList.add('has-result');
  contextDrawer.hidden = false;
  document.querySelector('.context-toggle').hidden = false;
  copyBtn.disabled = false;
  const citations = data.citations || [];
  const debug = data.retrieval_debug || [];
  const strongCount = debug.filter(item => item.evidence_grade === 'strong').length;
  const level = data.answer_type === 'grounded' ? 'low' : (data.answer_type === 'clarify' ? 'medium' : 'high');
  summarySlot.innerHTML = `<div class="answer-status"><div class="answer-status-main"><span class="status-chip" data-level="${level}">${answerTypeLabel(data.answer_type)}</span><span class="answer-status-label">${data.answer_type === 'grounded' ? '回答已由可访问资料支持' : '请结合证据边界确认回答'}</span></div><div class="answer-metrics"><span class="answer-metric"><strong>置信度</strong>${pct(data.confidence)}%</span><span class="answer-metric"><strong>风险</strong>${riskLabel(data.hallucination_risk)}</span><span class="answer-metric"><strong>引用</strong>${citations.length}</span><span class="answer-metric"><strong>强证据</strong>${strongCount}/${debug.length}</span></div></div>`;
  const refusal = data.answer_type === 'refusal' ? '<div class="guardrail-notice"><strong>回答已受到证据和权限边界保护</strong><div>系统没有扩展未被当前身份支持的内容，也没有暴露受限资料。</div></div>' : '';
  const inlineEvidence = citations.length ? `<section class="evidence-list"><strong>引用依据</strong>${citations.slice(0, 2).map(citation => `<blockquote class="evidence-quote"><span>${escapeHtml(citation.title)} · ${escapeHtml(citation.chunk_id)}</span><p>${escapeHtml(citation.quote)}</p></blockquote>`).join('')}</section>` : '';
  const evidence = data.evidence_summary ? `<div class="citation"><strong>证据摘要</strong><div>${escapeHtml(data.evidence_summary)}</div></div>` : '';
  const followUps = (data.follow_up_questions || []).length ? `<div class="citation"><strong>建议追问</strong><div class="follow-up-actions">${data.follow_up_questions.map(question => `<button class="chip" type="button" data-q="${escapeHtml(question)}">${escapeHtml(question)}</button>`).join('')}</div></div>` : '';
  answerTab.innerHTML = `${refusal}<div class="answer">${escapeHtml(data.answer)}</div>${inlineEvidence}${evidence}${followUps}`;
  citationsTab.innerHTML = citations.length ? citations.map(citation => `<div class="citation"><strong>${escapeHtml(citation.title)}</strong><div>${escapeHtml(citation.chunk_id)}</div><p>${escapeHtml(citation.quote)}</p></div>`).join('') : '<div class="empty">没有可用引用。</div>';
  traceSlot.innerHTML = debug.length ? debug.map(item => `<div class="trace-row ${item.evidence_grade === 'strong' ? 'strong' : ''}"><div class="trace-head"><span>${item.evidence_grade === 'strong' ? '强证据' : '弱证据'}</span><span>得分 ${escapeHtml(item.score)}</span></div><div class="trace-meta">${escapeHtml(item.source)} · ${escapeHtml(item.chunk_id)}</div><div class="trace-meta">命中原因：${escapeHtml((item.reasons || []).join(', '))}</div><details><summary class="advanced-summary">查看技术分数</summary><div class="score-bars"><div class="score-line"><span>BM25</span><span class="score-track"><span class="score-fill" style="width:${pct(item.bm25)}%"></span></span><span>${pct(item.bm25)}%</span></div><div class="score-line"><span>Vector</span><span class="score-track"><span class="score-fill vector" style="width:${pct(item.vector)}%"></span></span><span>${pct(item.vector)}%</span></div><div class="score-line"><span>Rerank</span><span class="score-track"><span class="score-fill rerank" style="width:${pct(item.rerank)}%"></span></span><span>${pct(item.rerank)}%</span></div></div></details></div>`).join('') : '<div class="empty">没有检索记录。</div>';
  document.getElementById('welcomeMessage').hidden = true;
  renderInspector(data);
  switchTab('answer');
}

function renderUploadSuccess(data) {
  showView('knowledge');
  summarySlot.innerHTML = '<div class="empty response-empty">资料库已更新，可以回到工作台开始提问。</div>';
  inspectorSlot.innerHTML = '<div class="inspector-empty"><strong>资料库已更新</strong><p>新文档已加入下一次混合检索的候选集合。</p></div>';
  statsPill.textContent = `${data.stats.documents} 份文档 / ${data.stats.chunks} 个片段`;
  knowledgeNotice.hidden = false;
  knowledgeNotice.innerHTML = `<strong>索引已完成。</strong> 已新增 ${data.added} 个可检索片段。建议回到问答工作台，用一个实际业务问题验证答案和引用。<br><button class="secondary compact" type="button" data-view="workspace">去验证资料</button>`;
}

function renderEval(data) {
  const leakLevel = data.permission_leaks > 0 ? 'high' : 'low';
  const readyForRelease = data.permission_leaks === 0 && Number(data.faithfulness) >= .8 && Number(data.citation_accuracy) >= .8;
  const recommendation = readyForRelease
    ? '建议发布：权限边界安全，回答忠实度和引用准确率达到当前发布门槛。'
    : '建议暂缓发布：请优先处理权限泄漏，或提升引用准确率和忠实度后再发布。';
  evaluationSlot.classList.remove('evaluation-empty');
  evaluationSlot.innerHTML = `<div class="eval-grid"><div class="eval-card"><div class="eval-label">Recall@K</div><div class="eval-value">${pct(data.recall_at_k)}%</div><div class="eval-note">预期来源是否被找回</div></div><div class="eval-card"><div class="eval-label">MRR</div><div class="eval-value">${pct(data.mrr)}%</div><div class="eval-note">正确来源的排序质量</div></div><div class="eval-card"><div class="eval-label">引用准确率</div><div class="eval-value">${pct(data.citation_accuracy)}%</div><div class="eval-note">来源引用是否正确</div></div><div class="eval-card"><div class="eval-label">忠实度</div><div class="eval-value">${pct(data.faithfulness)}%</div><div class="eval-note">回答是否受证据支持</div></div><div class="eval-card"><div class="eval-label">权限泄漏</div><div class="eval-value">${data.permission_leaks}</div><div class="eval-note">不应可见来源的命中数</div></div></div><div class="citation"><strong>${readyForRelease ? '可以发布' : '需要处理后再发布'}</strong><div>${recommendation} 本次共回放 ${data.total} 个案例。</div></div><div class="scenario-list">${Object.entries(data.scenario_summary || {}).map(([name, item]) => `<span class="pill" data-level="${item.permission_leaks ? 'high' : 'low'}">${escapeHtml(name)} ${item.passed}/${item.total}</span>`).join('')}</div><section class="inspector-section"><h3>案例明细</h3>${(data.cases || []).map((item, index) => `<div class="case-row"><div class="case-title">${index + 1}. ${escapeHtml(item.question)}</div><div class="case-meta"><span class="pill" data-level="${item.recall_hit ? 'low' : 'high'}">召回 ${item.recall_hit ? '命中' : '未命中'}</span><span class="pill" data-level="${item.permission_leak ? 'high' : 'low'}">权限 ${item.permission_leak ? '泄漏' : '安全'}</span><span class="pill">排名 ${item.first_rank || '-'}</span></div><div class="trace-meta">检索：${escapeHtml((item.retrieved_sources || []).join(', ') || '-')}</div></div>`).join('')}</section>`;
  evaluationSlot.dataset.level = leakLevel;
}

uploadFormEl.addEventListener('submit', async event => {
  event.preventDefault();
  setMode('uploading'); setBusy(true);
  try {
    const data = await parseApiResponse(await apiFetch('/upload', { method: 'POST', body: new FormData(uploadFormEl) }));
    renderUploadSuccess(data);
    uploadFormEl.reset();
    await refreshStats(); await refreshDocuments();
    setMode('ready');
  } catch (error) { renderError('导入失败', error.message || String(error)); setMode('error'); }
  finally { setBusy(false); }
});

askBtn.addEventListener('click', async () => {
  if (!questionInput.value.trim()) { renderError('无法提问', '问题不能为空。'); setMode('error'); return; }
  showView('workspace'); setMode('retrieving'); setBusy(true);
  renderLoading('正在检索证据', '先确认权限，再召回、重排并校验引用。');
  try {
    const data = await parseApiResponse(await apiFetch('/ask', { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ question: questionInput.value, user: userInput.value, roles: rolesInput.value.split(',').map(value => value.trim()).filter(Boolean), session_id: sessionId }) }));
    lastAnswer = data;
    await saveCurrentTurn(questionInput.value.trim(), data);
    renderConversationHistory(serverIdentityEnabled ? { turns: sessionTurns } : currentLocalSession());
    renderResult(data);
    setMode(data.answer_type || 'ready');
  } catch (error) { renderError('提问失败', error.message || String(error)); setMode('error'); }
  finally { setBusy(false); }
});

evalBtn.addEventListener('click', async () => {
  showView('evaluation'); setMode('evaluating'); setBusy(true);
  evaluationSlot.className = 'evaluation-empty';
  evaluationSlot.innerHTML = '<strong>正在运行评测</strong><p>回放固定问题集并检查检索、引用和权限边界。</p>';
  try { const data = await parseApiResponse(await apiFetch('/eval', { method: 'POST' })); renderEval(data); setMode('eval ready'); }
  catch (error) { evaluationSlot.innerHTML = `<strong>评测失败</strong><p>${escapeHtml(error.message || String(error))}</p>`; setMode('error'); }
  finally { setBusy(false); }
});

copyBtn.addEventListener('click', async () => {
  if (!lastAnswer) return;
  await navigator.clipboard.writeText(JSON.stringify(lastAnswer, null, 2));
  setMode('copied'); setTimeout(() => setMode(lastAnswer.answer_type || 'ready'), 900);
});

shareSessionBtn.addEventListener('click', async () => {
  if (!serverIdentityEnabled || !sessionTurns.length) return;
  const entered = window.prompt('输入协作者账号，多个账号用逗号分隔');
  if (entered === null) return;
  try {
    const users = entered.split(',').map(value => value.trim()).filter(Boolean);
    await parseApiResponse(await apiFetch(`/sessions/${encodeURIComponent(sessionId)}/share`, { method: 'POST', headers: { 'content-type': 'application/json' }, body: JSON.stringify({ users }) }));
    setMode('已共享');
    setTimeout(() => setMode(lastAnswer?.answer_type || 'ready'), 900);
  } catch (error) { renderError('共享失败', error.message || String(error)); setMode('error'); }
});

newSessionBtn.addEventListener('click', () => {
  sessionId = Math.random().toString(36).slice(2); sessionTurns = []; sessionOwner = serverIdentityEnabled ? userInput.value : ''; lastAnswer = null; questionInput.value = ''; resizeQuestionInput(); copyBtn.disabled = true; shareSessionBtn.hidden = !serverIdentityEnabled; renderSession(); showView('workspace');
  document.body.classList.remove('has-answer');
  document.querySelector('.response-area').classList.remove('has-result');
  contextDrawer.hidden = true;
  document.querySelector('.context-toggle').hidden = true;
  document.getElementById('welcomeMessage').hidden = false;
  conversationHistory.hidden = true;
  conversationHistory.innerHTML = '';
  summarySlot.innerHTML = '<div class="empty response-empty">新会话已就绪。提出一个业务问题开始吧。</div>';
  answerTab.innerHTML = '<div class="empty">等待问题。</div>'; citationsTab.innerHTML = '<div class="empty">暂无引用。</div>'; traceSlot.innerHTML = '<div class="empty">还没有检索记录。</div>';
  inspectorSlot.innerHTML = '<div class="inspector-empty"><strong>等待检索</strong><p>完成提问后，这里会显示权限边界、命中来源和证据强度。</p></div>'; setMode('ready');
});

document.addEventListener('click', async event => {
  const drawerButton = event.target.closest('[data-drawer]');
  if (drawerButton) { openDrawer(drawerButton.dataset.drawer); return; }
  if (event.target.closest('[data-close-drawer]')) { closeDrawers(); return; }
  const viewButton = event.target.closest('[data-view]');
  if (viewButton) { showView(viewButton.dataset.view); return; }
  if (event.target.closest('[data-new-session]')) { newSessionBtn.click(); return; }
  const questionButton = event.target.closest('[data-q]');
  if (questionButton) { questionInput.value = questionButton.getAttribute('data-q') || ''; resizeQuestionInput(); showView('workspace'); questionInput.focus(); return; }
  const localSessionButton = event.target.closest('[data-local-session]');
  if (localSessionButton) { restoreLocalSession(localSessionButton.getAttribute('data-local-session') || ''); return; }
  const serverSessionButton = event.target.closest('[data-server-session]');
  if (serverSessionButton) { await restoreServerSession(serverSessionButton.getAttribute('data-server-session') || ''); return; }
  if (event.target.closest('[data-clear-history]')) {
    if (serverIdentityEnabled) return;
    window.localStorage.removeItem(localSessionsKey); renderLocalSessions(); return;
  }
  const tab = event.target.closest('[data-tab]');
  if (tab) { switchTab(tab.dataset.tab); return; }
  const deleteButton = event.target.closest('[data-delete-doc]');
  if (deleteButton) {
    if (!confirm('删除这份文档及其所有片段？')) return;
    deleteButton.disabled = true;
    try { await parseApiResponse(await apiFetch(`/documents?id=${encodeURIComponent(deleteButton.dataset.deleteDoc)}`, { method: 'DELETE' })); await refreshStats(); await refreshDocuments(); setMode('ready'); }
    catch (error) { renderError('删除失败', error.message || String(error)); setMode('error'); }
  }
});

document.addEventListener('keydown', event => {
  if ((event.ctrlKey || event.metaKey) && event.key === 'Enter' && !askBtn.disabled) askBtn.click();
  if (event.key === 'Escape') closeDrawers();
});

refreshDocsBtn.addEventListener('click', () => refreshDocuments().catch(error => renderError('加载文档失败', error.message || String(error))));
userInput.addEventListener('input', () => { updateIdentitySummary(); saveLocalIdentity(); });
rolesInput.addEventListener('input', () => { updateIdentitySummary(); saveLocalIdentity(); });
questionInput.addEventListener('input', resizeQuestionInput);
async function initializeApp() {
  loadLocalIdentity(); renderSession(); setMode('ready');
  updateIdentitySummary(); resizeQuestionInput(); renderLocalSessions();
  await loadServerIdentity();
  refreshStats().catch(error => renderError('健康检查失败', error.message || String(error)));
  if (managementEnabled) refreshDocuments().catch(error => renderError('加载文档失败', error.message || String(error)));
}

initializeApp();
