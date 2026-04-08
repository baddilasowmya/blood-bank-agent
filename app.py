"""
Blood Bank Supply Agent – FastAPI Server with Interactive Dashboard
"""
from __future__ import annotations

import asyncio
import math
from collections import deque
from typing import Dict, List, Optional, Any

from fastapi import FastAPI, Body
from fastapi.responses import HTMLResponse, JSONResponse
from pydantic import BaseModel

from environment import (
    BloodBankEnvironment,
    BloodObservation,
    DeliveryAction,
    ActionType,
    Direction,
    UrgencyLevel,
    ZoneType,
    BLOOD_TYPES,
    SCENARIOS,
    get_scenario,
)

app = FastAPI(title="Blood Bank Supply Agent", version="1.0.0")

# Global environment instance
_env: BloodBankEnvironment = BloodBankEnvironment("city_shortage", 42)
_last_obs: Optional[BloodObservation] = None


# ---------------------------------------------------------------------------
# Request models
# ---------------------------------------------------------------------------

class ResetRequest(BaseModel):
    scenario: Optional[str] = "city_shortage"
    seed: Optional[int] = 42


class StepRequest(BaseModel):
    action: DeliveryAction


# ---------------------------------------------------------------------------
# HTML Dashboard
# ---------------------------------------------------------------------------

DASHBOARD_HTML = r"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8"/>
<meta name="viewport" content="width=device-width, initial-scale=1.0"/>
<title>Blood Bank Supply Agent</title>
<style>
  * { box-sizing: border-box; margin: 0; padding: 0; }
  body { background: #0d1117; color: #e6edf3; font-family: 'Segoe UI', system-ui, sans-serif; min-height: 100vh; }
  header { background: linear-gradient(135deg, #8b0000 0%, #1a0000 100%); padding: 14px 24px; display: flex; align-items: center; justify-content: space-between; border-bottom: 2px solid #b22222; }
  header h1 { font-size: 1.4rem; color: #ff6b6b; letter-spacing: 1px; }
  .header-info { display: flex; gap: 20px; align-items: center; font-size: 0.85rem; }
  .badge { background: #161b22; padding: 4px 12px; border-radius: 20px; border: 1px solid #30363d; }
  .badge span { color: #ff8080; font-weight: 700; }
  .main { display: grid; grid-template-columns: 260px 1fr 280px; gap: 12px; padding: 12px; height: calc(100vh - 60px); }
  .panel { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 12px; overflow-y: auto; }
  .panel h3 { color: #ff8080; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 1px; margin-bottom: 10px; border-bottom: 1px solid #30363d; padding-bottom: 6px; }
  .center-col { display: flex; flex-direction: column; gap: 10px; }
  #grid-container { background: #0d1117; border: 1px solid #30363d; border-radius: 8px; padding: 10px; flex: 1; display: flex; align-items: center; justify-content: center; }
  #grid { display: grid; grid-template-columns: repeat(10, 52px); grid-template-rows: repeat(10, 52px); gap: 2px; }
  .cell { width: 52px; height: 52px; border-radius: 4px; display: flex; flex-direction: column; align-items: center; justify-content: center; cursor: default; font-size: 9px; border: 1px solid #30363d; transition: transform 0.1s; position: relative; }
  .cell:hover { transform: scale(1.05); z-index: 10; }
  .cell.empty    { background: #161b22; }
  .cell.blocked  { background: #1c1c1c; border-color: #444; color: #555; }
  .cell.blood-bank   { background: #0d2b45; border-color: #1e6091; color: #4fc3f7; }
  .cell.donor-center { background: #0d3320; border-color: #1e7e45; color: #66bb6a; }
  .cell.hospital-critical { background: #3d0000; border-color: #cc0000; color: #ff6b6b; animation: pulse-critical 1.5s infinite; }
  .cell.hospital-high     { background: #2d1500; border-color: #e65100; color: #ff8a50; }
  .cell.hospital-moderate { background: #1a1a00; border-color: #827717; color: #d4e157; }
  .cell.hospital-low      { background: #0d1a1a; border-color: #006064; color: #80cbc4; }
  .cell.hospital-stable   { background: #161b22; border-color: #30363d; color: #8b949e; }
  .cell.agent { outline: 3px solid #ffd700; outline-offset: 1px; }
  @keyframes pulse-critical { 0%,100% { box-shadow: 0 0 6px #cc0000; } 50% { box-shadow: 0 0 14px #ff0000; } }
  .cell-icon { font-size: 18px; line-height: 1; }
  .cell-label { font-size: 8px; text-align: center; max-width: 46px; overflow: hidden; white-space: nowrap; text-overflow: ellipsis; }
  .cell-coord { font-size: 7px; color: #444; position: absolute; top: 2px; left: 3px; }
  select, button { background: #21262d; color: #e6edf3; border: 1px solid #30363d; border-radius: 6px; padding: 6px 10px; font-size: 0.8rem; width: 100%; margin-bottom: 8px; cursor: pointer; }
  button { background: #8b0000; border-color: #cc0000; color: #fff; font-weight: 600; transition: background 0.2s; }
  button:hover { background: #b22222; }
  button.secondary { background: #21262d; border-color: #30363d; color: #8b949e; }
  button.secondary:hover { background: #30363d; color: #e6edf3; }
  .stat-row { display: flex; justify-content: space-between; align-items: center; padding: 4px 0; border-bottom: 1px solid #21262d; font-size: 0.78rem; }
  .stat-row .val { font-weight: 700; color: #ff8080; }
  .stat-row .val.green { color: #66bb6a; }
  .stat-row .val.yellow { color: #ffd700; }
  .stat-row .val.red { color: #f44336; }
  .inv-grid { display: grid; grid-template-columns: 1fr 1fr; gap: 4px; margin-top: 6px; }
  .inv-item { background: #21262d; border-radius: 4px; padding: 5px 8px; display: flex; justify-content: space-between; font-size: 0.75rem; border-left: 3px solid #8b0000; }
  .inv-item .bt { color: #ff8080; font-weight: 700; }
  .inv-item .qty { color: #e6edf3; }
  .inv-item.zero { border-left-color: #30363d; opacity: 0.5; }
  .hospital-item { padding: 6px; margin-bottom: 6px; border-radius: 6px; font-size: 0.75rem; border-left: 3px solid #30363d; background: #0d1117; }
  .hospital-item.critical { border-left-color: #cc0000; background: #1a0000; }
  .hospital-item.high     { border-left-color: #e65100; background: #160900; }
  .hospital-item.moderate { border-left-color: #827717; background: #0d0d00; }
  .hospital-item.low      { border-left-color: #006064; background: #001010; }
  .hospital-name { font-weight: 700; color: #e6edf3; margin-bottom: 3px; }
  .hospital-needs { color: #8b949e; font-size: 0.7rem; }
  .hospital-needs span { color: #ff8080; }
  .log-entry { padding: 4px 6px; margin-bottom: 4px; background: #0d1117; border-radius: 4px; font-size: 0.72rem; border-left: 2px solid #30363d; color: #8b949e; }
  .log-entry.success { border-left-color: #66bb6a; color: #66bb6a; }
  .log-entry.error   { border-left-color: #f44336; color: #f44336; }
  .log-entry.info    { border-left-color: #4fc3f7; color: #4fc3f7; }
  .progress-bar { height: 6px; background: #21262d; border-radius: 3px; overflow: hidden; margin: 3px 0; }
  .progress-fill { height: 100%; background: linear-gradient(90deg, #8b0000, #e53935); transition: width 0.3s; }
  .progress-fill.green { background: linear-gradient(90deg, #1b5e20, #66bb6a); }
  #controls { background: #161b22; border: 1px solid #30363d; border-radius: 8px; padding: 10px; }
  .dir-grid { display: grid; grid-template-columns: 1fr 1fr 1fr; gap: 4px; margin: 6px 0; }
  .dir-grid button { margin: 0; padding: 4px; font-size: 0.7rem; }
  .dir-grid .center-btn { grid-column: 2; }
  .action-row { display: flex; gap: 6px; margin-bottom: 6px; }
  .action-row select { margin: 0; flex: 1; }
  .action-row input { background: #21262d; color: #e6edf3; border: 1px solid #30363d; border-radius: 6px; padding: 6px 8px; font-size: 0.8rem; width: 70px; }
  .action-row button { margin: 0; flex: 0 0 80px; }
  label { font-size: 0.75rem; color: #8b949e; display: block; margin-bottom: 3px; }
  #toast { position: fixed; top: 50%; left: 50%; transform: translate(-50%,-50%) scale(0); background: linear-gradient(135deg, #1b5e20, #2e7d32); border: 2px solid #66bb6a; border-radius: 12px; padding: 30px 40px; text-align: center; z-index: 1000; transition: transform 0.3s; box-shadow: 0 20px 60px rgba(0,0,0,0.8); }
  #toast.show { transform: translate(-50%,-50%) scale(1); }
  #toast h2 { color: #a5d6a7; font-size: 1.5rem; margin-bottom: 8px; }
  #toast p  { color: #e6edf3; font-size: 0.9rem; }
  .legend { display: flex; flex-wrap: wrap; gap: 6px; margin-bottom: 8px; font-size: 0.7rem; }
  .legend-item { display: flex; align-items: center; gap: 4px; }
  .legend-dot { width: 10px; height: 10px; border-radius: 2px; }
  #score-display { text-align: center; padding: 6px; background: #0d1117; border-radius: 6px; margin-bottom: 8px; }
  #score-display .score-val { font-size: 2rem; font-weight: 700; color: #ffd700; }
  #score-display .score-label { font-size: 0.7rem; color: #8b949e; }
  .steps-bar { display: flex; align-items: center; gap: 8px; font-size: 0.75rem; }
  .steps-bar .pb { flex: 1; }
  .steps-bar .num { color: #8b949e; white-space: nowrap; }
  .urgency-badge { display: inline-block; padding: 1px 6px; border-radius: 10px; font-size: 0.65rem; font-weight: 700; text-transform: uppercase; }
  .urgency-badge.critical { background: #3d0000; color: #ff6b6b; }
  .urgency-badge.high     { background: #2d1500; color: #ff8a50; }
  .urgency-badge.moderate { background: #1a1a00; color: #d4e157; }
  .urgency-badge.low      { background: #0d1a1a; color: #80cbc4; }
  .urgency-badge.stable   { background: #161b22; color: #8b949e; }
  ::-webkit-scrollbar { width: 4px; } ::-webkit-scrollbar-track { background: #0d1117; } ::-webkit-scrollbar-thumb { background: #30363d; border-radius: 2px; }
</style>
</head>
<body>
<header>
  <h1>🩸 Blood Bank Supply Agent</h1>
  <div class="header-info">
    <div class="badge">Step: <span id="hdr-step">0</span> / <span id="hdr-max">70</span></div>
    <div class="badge">Scenario: <span id="hdr-scenario">city_shortage</span></div>
    <div class="badge">Agent: (<span id="hdr-ax">5</span>,<span id="hdr-ay">5</span>)</div>
    <div class="badge">Lives Saved: <span id="hdr-pct">0</span>%</div>
  </div>
</header>

<div class="main">
  <!-- Left Panel -->
  <div class="panel" id="left-panel">
    <h3>Scenario</h3>
    <label for="scenario-sel">Select Scenario</label>
    <select id="scenario-sel">
      <option value="city_shortage">Easy – Mumbai City Shortage</option>
      <option value="rare_type_emergency">Medium – Delhi Rare Blood Emergency</option>
      <option value="disaster_response">Hard – Chennai Disaster Response</option>
    </select>
    <button onclick="resetEnv()">↺ Reset Environment</button>

    <h3 style="margin-top:12px;">Score</h3>
    <div id="score-display">
      <div class="score-val" id="score-val">0.00</div>
      <div class="score-label">Composite Score (0–1)</div>
    </div>

    <h3>Statistics</h3>
    <div class="stat-row"><span>Total Patients</span><span class="val" id="stat-total">0</span></div>
    <div class="stat-row"><span>Saved</span><span class="val green" id="stat-saved">0</span></div>
    <div class="stat-row"><span>Lost</span><span class="val red" id="stat-lost">0</span></div>
    <div class="stat-row"><span>Lives Saved %</span><span class="val yellow" id="stat-pct">0%</span></div>
    <div class="stat-row"><span>Last Reward</span><span class="val" id="stat-reward">0.00</span></div>
    <div class="progress-bar" style="margin:6px 0"><div class="progress-fill green" id="lives-bar" style="width:0%"></div></div>

    <h3 style="margin-top:10px;">Agent Inventory</h3>
    <div class="stat-row"><span>Capacity Remaining</span><span class="val" id="stat-cap">100</span></div>
    <div class="inv-grid" id="inv-grid"></div>

    <h3 style="margin-top:10px;">Steps</h3>
    <div class="steps-bar">
      <div class="pb"><div class="progress-bar"><div class="progress-fill" id="steps-bar" style="width:0%"></div></div></div>
      <div class="num"><span id="steps-num">0</span>/<span id="steps-max">70</span></div>
    </div>
  </div>

  <!-- Center -->
  <div class="center-col">
    <!-- Controls -->
    <div id="controls">
      <h3 style="color:#ff8080;font-size:0.8rem;text-transform:uppercase;letter-spacing:1px;margin-bottom:8px;">Controls</h3>
      <div style="display:flex;gap:10px;">
        <div style="flex:0 0 140px;">
          <label>Movement</label>
          <div class="dir-grid">
            <div></div>
            <button onclick="move('north')">▲ N</button>
            <div></div>
            <button onclick="move('west')">◄ W</button>
            <button class="secondary center-btn" onclick="doWait()">Wait</button>
            <button onclick="move('east')">E ►</button>
            <div></div>
            <button onclick="move('south')">▼ S</button>
            <div></div>
          </div>
        </div>
        <div style="flex:1;">
          <label>Deliver Blood</label>
          <div class="action-row">
            <select id="del-zone"><option value="">Zone…</option></select>
            <select id="del-bt"><option value="">Type…</option></select>
            <input type="number" id="del-qty" value="10" min="1" max="50" placeholder="Qty"/>
            <button onclick="deliverBlood()">Deliver</button>
          </div>
          <label>Collect Blood</label>
          <div class="action-row">
            <select id="col-zone"><option value="">Zone…</option></select>
            <select id="col-bt"><option value="">Type…</option></select>
            <input type="number" id="col-qty" value="20" min="1" max="50" placeholder="Qty"/>
            <button onclick="collectBlood()" style="background:#1b5e20;border-color:#388e3c;">Collect</button>
          </div>
        </div>
      </div>
    </div>

    <!-- Grid -->
    <div id="grid-container">
      <div>
        <div class="legend">
          <div class="legend-item"><div class="legend-dot" style="background:#cc0000"></div>Critical Hospital</div>
          <div class="legend-item"><div class="legend-dot" style="background:#e65100"></div>High</div>
          <div class="legend-item"><div class="legend-dot" style="background:#827717"></div>Moderate</div>
          <div class="legend-item"><div class="legend-dot" style="background:#1e6091"></div>Blood Bank</div>
          <div class="legend-item"><div class="legend-dot" style="background:#1e7e45"></div>Donor Center</div>
          <div class="legend-item"><div class="legend-dot" style="background:#ffd700;border:1px solid #ffd700"></div>Agent</div>
          <div class="legend-item"><div class="legend-dot" style="background:#333"></div>Blocked</div>
        </div>
        <div id="grid"></div>
      </div>
    </div>
  </div>

  <!-- Right Panel -->
  <div class="panel" id="right-panel">
    <h3>Hospitals</h3>
    <div id="hospital-list"></div>

    <h3 style="margin-top:10px;">Action Log</h3>
    <div id="action-log"></div>
  </div>
</div>

<!-- Mission Toast -->
<div id="toast">
  <h2>🎉 Mission Complete!</h2>
  <p id="toast-msg">All critical patients have been served.</p>
  <button onclick="document.getElementById('toast').classList.remove('show')" style="margin-top:12px;width:auto;padding:6px 20px;">Close</button>
</div>

<script>
const BLOOD_TYPES = ['O+','O-','A+','A-','B+','B-','AB+','AB-'];
let state = null;
let logEntries = [];

async function api(path, method='GET', body=null) {
  const opts = { method, headers: {'Content-Type':'application/json'} };
  if (body) opts.body = JSON.stringify(body);
  const r = await fetch(path, opts);
  return r.json();
}

async function resetEnv() {
  const sc = document.getElementById('scenario-sel').value;
  const data = await api('/reset','POST',{scenario:sc,seed:42});
  state = data;
  logEntries = [];
  addLog('Environment reset: '+sc,'info');
  updateUI();
}

async function move(dir) {
  const data = await api('/step','POST',{action:{action_type:'move',direction:dir}});
  handleStepResponse(data);
}

async function doWait() {
  const data = await api('/step','POST',{action:{action_type:'wait'}});
  handleStepResponse(data);
}

async function deliverBlood() {
  const zone = document.getElementById('del-zone').value;
  const bt = document.getElementById('del-bt').value;
  const qty = parseInt(document.getElementById('del-qty').value)||10;
  if (!zone||!bt) { addLog('Select zone and blood type','error'); return; }
  const data = await api('/step','POST',{action:{action_type:'deliver',target_zone_id:zone,blood_type:bt,quantity:qty}});
  handleStepResponse(data);
}

async function collectBlood() {
  const zone = document.getElementById('col-zone').value;
  const bt = document.getElementById('col-bt').value;
  const qty = parseInt(document.getElementById('col-qty').value)||20;
  if (!zone||!bt) { addLog('Select zone and blood type','error'); return; }
  const data = await api('/step','POST',{action:{action_type:'collect',target_zone_id:zone,blood_type:bt,quantity:qty}});
  handleStepResponse(data);
}

function handleStepResponse(data) {
  if (!data || data.detail) { addLog('Error: '+(data?.detail||'unknown'),'error'); return; }
  state = data.observation || data;
  const msg = state.last_action_result||'';
  const reward = state.last_reward||0;
  const cls = reward>0?'success':reward<-0.15?'error':'info';
  addLog(`[${state.step_number}] ${msg} (r=${reward.toFixed(2)})`, cls);
  updateUI();
  if (state.is_complete && state.mission_success) showToast(state);
}

function addLog(msg, cls='info') {
  logEntries.unshift({msg,cls});
  if (logEntries.length>50) logEntries.pop();
  renderLog();
}

function renderLog() {
  const el = document.getElementById('action-log');
  el.innerHTML = logEntries.map(e=>`<div class="log-entry ${e.cls}">${e.msg}</div>`).join('');
}

function updateUI() {
  if (!state) return;
  const obs = state.observation || state;

  // Header
  document.getElementById('hdr-step').textContent = obs.step_number||0;
  document.getElementById('hdr-max').textContent  = obs.max_steps||70;
  document.getElementById('hdr-scenario').textContent = obs.scenario_name||'';
  const ag = obs.agent||{};
  document.getElementById('hdr-ax').textContent = ag.x??'?';
  document.getElementById('hdr-ay').textContent = ag.y??'?';
  const pct = obs.lives_saved_pct||0;
  document.getElementById('hdr-pct').textContent = pct.toFixed(1);

  // Stats
  document.getElementById('stat-total').textContent  = obs.total_patients||0;
  document.getElementById('stat-saved').textContent  = obs.patients_saved||0;
  document.getElementById('stat-lost').textContent   = obs.patients_lost||0;
  document.getElementById('stat-pct').textContent    = pct.toFixed(1)+'%';
  document.getElementById('stat-reward').textContent = (obs.last_reward||0).toFixed(2);
  document.getElementById('lives-bar').style.width   = Math.min(100,pct)+'%';
  document.getElementById('stat-cap').textContent    = ag.capacity_remaining??'?';

  const step = obs.step_number||0;
  const maxS = obs.max_steps||70;
  document.getElementById('steps-bar').style.width = Math.min(100,step/maxS*100)+'%';
  document.getElementById('steps-num').textContent = step;
  document.getElementById('steps-max').textContent = maxS;

  // Score (async fetch)
  fetch('/grader').then(r=>r.json()).then(d=>{
    document.getElementById('score-val').textContent=(d.score||0).toFixed(3);
  }).catch(()=>{});

  // Inventory
  const inv = ag.inventory||{};
  const invGrid = document.getElementById('inv-grid');
  invGrid.innerHTML = BLOOD_TYPES.map(bt=>{
    const qty = inv[bt]||0;
    return `<div class="inv-item ${qty===0?'zero':''}"><span class="bt">${bt}</span><span class="qty">${qty}</span></div>`;
  }).join('');

  // Grid
  buildGrid(obs);

  // Hospitals
  buildHospitalList(obs);

  // Populate action dropdowns
  populateDropdowns(obs);
}

function getZoneClass(z, agX, agY) {
  let cls = '';
  if (z.zone_type==='blocked') cls='blocked';
  else if (z.zone_type==='blood_bank') cls='blood-bank';
  else if (z.zone_type==='donor_center') cls='donor-center';
  else if (z.zone_type==='hospital') cls=`hospital-${z.urgency||'stable'}`;
  else cls='empty';
  if (z.x===agX && z.y===agY) cls+=' agent';
  return cls;
}

function zoneIcon(z, agX, agY) {
  if (z.x===agX && z.y===agY) return '🚑';
  if (z.zone_type==='blocked') return '🚧';
  if (z.zone_type==='blood_bank') return '🏦';
  if (z.zone_type==='donor_center') return '💉';
  if (z.zone_type==='hospital') {
    if (z.urgency==='critical') return '🆘';
    if (z.urgency==='high') return '🏥';
    return '🏥';
  }
  return '';
}

function buildGrid(obs) {
  const grid = document.getElementById('grid');
  const ag = obs.agent||{};
  const agX=ag.x??-1, agY=ag.y??-1;

  // Build lookup
  const zmap = {};
  (obs.zones||[]).forEach(z=>{ zmap[z.x+','+z.y]=z; });

  let html='';
  for(let y=0;y<10;y++){
    for(let x=0;x<10;x++){
      const z = zmap[x+','+y];
      if(!z){html+=`<div class="cell empty"><span class="cell-coord">${x},${y}</span></div>`;continue;}
      const cls=getZoneClass(z,agX,agY);
      const icon=zoneIcon(z,agX,agY);
      const label=z.zone_type==='empty'?'':z.name.split(' ').slice(0,2).join(' ');
      const needStr = z.zone_type==='hospital' ? Object.entries(z.needs||{}).filter(([,v])=>v>0).map(([k,v])=>`${k}:${v}`).join(' ') : '';
      const stockTotal = z.zone_type!=='hospital' ? Object.values(z.stock||{}).reduce((a,b)=>a+b,0) : 0;
      let extra='';
      if(z.zone_type==='hospital' && needStr) extra=`<span style="font-size:6px;color:#888">${needStr.slice(0,18)}</span>`;
      else if(stockTotal>0) extra=`<span style="font-size:7px;color:#555">${stockTotal}u</span>`;
      html+=`<div class="cell ${cls}" title="${z.name} (${x},${y})\n${needStr||stockTotal+' units'}">
        <span class="cell-coord">${x},${y}</span>
        <span class="cell-icon">${icon}</span>
        <span class="cell-label">${label}</span>
        ${extra}
      </div>`;
    }
  }
  grid.innerHTML=html;
}

function buildHospitalList(obs) {
  const hospitals = (obs.zones||[]).filter(z=>z.zone_type==='hospital');
  hospitals.sort((a,b)=>{
    const order={critical:0,high:1,moderate:2,low:3,stable:4};
    return (order[a.urgency]||4)-(order[b.urgency]||4);
  });
  const el=document.getElementById('hospital-list');
  el.innerHTML=hospitals.map(h=>{
    const needs=Object.entries(h.needs||{}).filter(([,v])=>v>0);
    const needStr=needs.map(([k,v])=>`<span>${k}:${v}</span>`).join(' ');
    return `<div class="hospital-item ${h.urgency||'stable'}">
      <div class="hospital-name">${h.name} <span class="urgency-badge ${h.urgency||'stable'}">${h.urgency||'stable'}</span></div>
      <div class="hospital-needs">Waiting: ${h.patients_waiting} | Saved: ${h.patients_saved}<br>Needs: ${needStr||'<span style="color:#66bb6a">Fulfilled</span>'}</div>
    </div>`;
  }).join('');
}

function populateDropdowns(obs) {
  const hospitals=(obs.zones||[]).filter(z=>z.zone_type==='hospital');
  const sources=(obs.zones||[]).filter(z=>z.zone_type==='blood_bank'||z.zone_type==='donor_center');

  const delZone=document.getElementById('del-zone');
  const prevDel=delZone.value;
  delZone.innerHTML='<option value="">Zone…</option>'+hospitals.map(z=>`<option value="${z.zone_id}">${z.name}</option>`).join('');
  if(prevDel) delZone.value=prevDel;

  const colZone=document.getElementById('col-zone');
  const prevCol=colZone.value;
  colZone.innerHTML='<option value="">Zone…</option>'+sources.map(z=>`<option value="${z.zone_id}">${z.name}</option>`).join('');
  if(prevCol) colZone.value=prevCol;

  [document.getElementById('del-bt'), document.getElementById('col-bt')].forEach(sel=>{
    const prev=sel.value;
    sel.innerHTML='<option value="">Type…</option>'+BLOOD_TYPES.map(bt=>`<option value="${bt}">${bt}</option>`).join('');
    if(prev) sel.value=prev;
  });
}

function showToast(obs) {
  document.getElementById('toast-msg').textContent=`Lives saved: ${(obs.lives_saved_pct||0).toFixed(1)}% in ${obs.step_number} steps.`;
  document.getElementById('toast').classList.add('show');
}

// Init
(async()=>{
  const data = await api('/state');
  if(data && !data.status) {
    // Fetch full obs from reset
    const resetData = await api('/reset','POST',{scenario:'city_shortage',seed:42});
    state = resetData;
    updateUI();
    addLog('Dashboard initialized','info');
  } else {
    const resetData = await api('/reset','POST',{scenario:'city_shortage',seed:42});
    state = resetData;
    updateUI();
    addLog('Dashboard initialized','info');
  }
})();
</script>
</body>
</html>
"""


# ---------------------------------------------------------------------------
# Endpoints
# ---------------------------------------------------------------------------

@app.get("/", response_class=HTMLResponse)
async def dashboard():
    return HTMLResponse(content=DASHBOARD_HTML)


@app.get("/health")
async def health():
    return {"status": "ok", "project": "Blood Bank Supply Agent"}


@app.post("/reset")
async def reset(req: Optional[ResetRequest] = Body(default=None)):
    global _env, _last_obs
    scenario = (req.scenario if req and req.scenario else None) or "city_shortage"
    seed = (req.seed if req and req.seed is not None else None) or 42
    _env = BloodBankEnvironment(scenario, seed)
    _last_obs = await _env.reset()
    return _last_obs


@app.post("/step")
async def step(req: StepRequest):
    global _last_obs
    obs, reward, done, info = await _env.step(req.action)
    _last_obs = obs
    return {"observation": obs, "reward": reward, "done": done, "info": info}


@app.get("/state")
async def get_state():
    return _env.state


@app.get("/tasks")
async def tasks():
    return {
        "tasks": [
            {
                "id": "easy",
                "scenario": "city_shortage",
                "display_name": "Mumbai City Blood Shortage",
                "difficulty": "easy",
                "max_steps": 70,
                "description": (
                    "Moderate shortage across 4 hospitals in Mumbai. "
                    "Blood banks are nearby with adequate stock. "
                    "Goal: deliver blood to prevent patient deterioration."
                ),
            },
            {
                "id": "medium",
                "scenario": "rare_type_emergency",
                "display_name": "Delhi Rare Blood Emergency",
                "difficulty": "medium",
                "max_steps": 55,
                "description": (
                    "Critical shortage of rare blood types (O-, AB-, B-) "
                    "across 7 hospitals in Delhi with blocked routes. "
                    "Blood banks are far from hospitals – careful routing required."
                ),
            },
            {
                "id": "hard",
                "scenario": "disaster_response",
                "display_name": "Chennai Disaster Response",
                "difficulty": "hard",
                "max_steps": 65,
                "description": (
                    "Post-disaster scenario with 8 hospitals, 3 critical, "
                    "12 blocked zones, and limited capacity. "
                    "Agent starts at the only nearby blood bank with minimal inventory."
                ),
            },
        ]
    }


@app.get("/grader")
async def grader():
    st = _env.state
    if st.get("status") == "not_initialized":
        return {"score": 0.01}

    lives_pct = st.get("lives_saved_pct", 0.01)
    step = st.get("step", 0)
    max_steps = st.get("max_steps", 70)

    # Utilization: how much of capacity was actively used
    cap_remaining = st.get("capacity_remaining", 100)
    # infer capacity from scenario
    scenario = st.get("scenario", "city_shortage")
    sc_data = SCENARIOS.get(scenario, {})
    capacity = sc_data.get("capacity", 100)
    utilization = max(0.0, 1.0 - cap_remaining / capacity) if capacity > 0 else 0.0

    # Speed bonus
    speed = max(0.0, 1.0 - step / max_steps) if max_steps > 0 else 0.0

    raw_score = 0.7 * (lives_pct / 100.0) + 0.15 * utilization + 0.15 * speed
    # Clamp strictly within (0, 1) as required by the evaluator
    score = round(max(0.01, min(0.99, raw_score)), 4)

    return {
        "score": score,
        "breakdown": {
            "lives_saved_pct": lives_pct,
            "utilization": round(utilization, 4),
            "speed": round(speed, 4),
            "weights": {"lives_saved": 0.70, "utilization": 0.15, "speed": 0.15},
        },
    }


@app.get("/baseline")
async def baseline():
    results = []
    for task_id, scenario in [
        ("easy", "city_shortage"),
        ("medium", "rare_type_emergency"),
        ("hard", "disaster_response"),
    ]:
        score, lives_pct, steps_used = await _run_greedy_baseline(scenario)
        results.append({
            "task_id": task_id,
            "scenario": scenario,
            "score": round(score, 4),
            "lives_saved_pct": round(lives_pct, 2),
            "steps_used": steps_used,
        })
    return {"baseline_results": results}


# ---------------------------------------------------------------------------
# Greedy Baseline Agent
# ---------------------------------------------------------------------------

async def _run_greedy_baseline(scenario_name: str) -> tuple:
    """Run a greedy agent and return (score, lives_pct, steps)."""
    env = BloodBankEnvironment(scenario_name, rng_seed=99)
    obs = await env.reset()
    done = False
    step_count = 0

    while not done:
        action = _greedy_action(obs)
        obs, reward, done, info = await env.step(action)
        step_count += 1

    st = env.state
    lives_pct = st.get("lives_saved_pct", 0.0)
    capacity = SCENARIOS.get(scenario_name, {}).get("capacity", 100)
    cap_remaining = st.get("capacity_remaining", capacity)
    utilization = max(0.0, 1.0 - cap_remaining / capacity)
    max_steps = obs.max_steps
    speed = max(0.0, 1.0 - step_count / max_steps)
    score = 0.7 * (lives_pct / 100.0) + 0.15 * utilization + 0.15 * speed
    # Clamp strictly within (0, 1)
    return round(max(0.01, min(0.99, score)), 4), lives_pct, step_count


def _greedy_action(obs: BloodObservation) -> DeliveryAction:
    """
    Greedy policy:
    1. If at hospital with needs and has compatible blood → deliver
    2. If inventory < 30% capacity → go to nearest blood source
    3. If at blood source with low inventory → collect most-needed blood type
    4. Otherwise → BFS navigate to nearest critical hospital
    """
    agent = obs.agent
    ax, ay = agent.x, agent.y
    inventory = agent.inventory
    inv_total = agent.total_units
    cap_remaining = agent.capacity_remaining
    capacity = inv_total + cap_remaining

    # Build zone lookup
    zone_map: Dict[str, Any] = {z.zone_id: z for z in obs.zones}
    current_zone_id = agent.current_zone_id
    current_zone = zone_map.get(current_zone_id)

    # Step 1: If at hospital, try to deliver
    if current_zone and current_zone.zone_type == ZoneType.hospital:
        for need_type, need_qty in current_zone.needs.items():
            if need_qty <= 0:
                continue
            # Find compatible blood in inventory
            from environment import COMPATIBILITY
            for donor_type in COMPATIBILITY.get(need_type, []):
                if inventory.get(donor_type, 0) > 0:
                    qty = min(inventory[donor_type], need_qty, 20)
                    return DeliveryAction(
                        action_type=ActionType.deliver,
                        target_zone_id=current_zone_id,
                        blood_type=donor_type,
                        quantity=qty,
                    )

    # Step 2: Check if low inventory
    low_inventory = inv_total < capacity * 0.30

    # Step 3: If at blood source and want to collect
    if current_zone and current_zone.zone_type in (ZoneType.blood_bank, ZoneType.donor_center):
        if low_inventory or cap_remaining > 20:
            # Find which blood type is most needed globally
            need_counts: Dict[str, int] = {}
            for z in obs.zones:
                if z.zone_type == ZoneType.hospital:
                    for bt, qty in z.needs.items():
                        need_counts[bt] = need_counts.get(bt, 0) + qty
            # Pick blood type available at this source that is most needed
            best_bt = None
            best_score = -1
            for bt, stock_qty in current_zone.stock.items():
                if stock_qty > 0:
                    score = need_counts.get(bt, 0)
                    if score > best_score:
                        best_score = score
                        best_bt = bt
            if best_bt is None:
                # Just collect whatever is available
                for bt in BLOOD_TYPES:
                    if current_zone.stock.get(bt, 0) > 0:
                        best_bt = bt
                        break
            if best_bt:
                qty = min(cap_remaining, current_zone.stock.get(best_bt, 0), 30)
                if qty > 0:
                    return DeliveryAction(
                        action_type=ActionType.collect,
                        target_zone_id=current_zone_id,
                        blood_type=best_bt,
                        quantity=qty,
                    )

    # Step 4: Navigate
    # Determine target
    target_zone = None

    if low_inventory:
        # Find nearest blood source with stock
        target_zone = _nearest_zone(
            ax, ay,
            [z for z in obs.zones
             if z.zone_type in (ZoneType.blood_bank, ZoneType.donor_center)
             and sum(z.stock.values()) > 0],
            obs,
        )
    else:
        # Go to nearest critical/high hospital with remaining needs
        priority_hospitals = sorted(
            [z for z in obs.zones
             if z.zone_type == ZoneType.hospital and sum(z.needs.values()) > 0],
            key=lambda z: (
                {"critical": 0, "high": 1, "moderate": 2, "low": 3, "stable": 4}.get(z.urgency.value, 4),
                abs(z.x - ax) + abs(z.y - ay),
            )
        )
        if priority_hospitals:
            target_zone = priority_hospitals[0]

    if target_zone is None:
        return DeliveryAction(action_type=ActionType.wait)

    # BFS to find next step toward target
    direction = _bfs_next_direction(ax, ay, target_zone.x, target_zone.y, obs)
    if direction:
        return DeliveryAction(action_type=ActionType.move, direction=direction)

    return DeliveryAction(action_type=ActionType.wait)


def _nearest_zone(ax: int, ay: int, zones: list, obs: BloodObservation):
    if not zones:
        return None
    return min(zones, key=lambda z: abs(z.x - ax) + abs(z.y - ay))


def _bfs_next_direction(ax: int, ay: int, tx: int, ty: int, obs: BloodObservation) -> Optional[Direction]:
    """BFS on 10x10 grid avoiding blocked zones, returns first direction to take."""
    if ax == tx and ay == ty:
        return None

    blocked_set = {(z.x, z.y) for z in obs.zones if z.zone_type == ZoneType.blocked}

    from collections import deque as _deque
    queue = _deque()
    queue.append((ax, ay, []))
    visited = {(ax, ay)}

    dir_map = {
        Direction.north: (0, -1),
        Direction.south: (0, 1),
        Direction.west:  (-1, 0),
        Direction.east:  (1, 0),
    }

    while queue:
        cx, cy, path = queue.popleft()
        for d, (ddx, ddy) in dir_map.items():
            nx, ny = cx + ddx, cy + ddy
            if not (0 <= nx < 10 and 0 <= ny < 10):
                continue
            if (nx, ny) in visited:
                continue
            if (nx, ny) in blocked_set:
                continue
            new_path = path + [d]
            if nx == tx and ny == ty:
                return new_path[0] if new_path else None
            visited.add((nx, ny))
            queue.append((nx, ny, new_path))

    # Fallback: move greedily
    if tx > ax:
        return Direction.east
    if tx < ax:
        return Direction.west
    if ty > ay:
        return Direction.south
    return Direction.north
