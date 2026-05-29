(function () {
  let range = 7, chart = null;

  const fmt = n => n == null ? '—' : Number(n).toLocaleString();
  const pct = n => n == null ? '—' : Number(n).toFixed(1) + '%';
  const esc = s => String(s).replace(/[&<>"']/g, c => ({'&':'&amp;','<':'&lt;','>':'&gt;','"':'&quot;',"'":'&#39;'}[c]));
  const ago = iso => { const s=(Date.now()-new Date(iso))/1000; return s<60?'Just now':s<3600?Math.floor(s/60)+'m ago':s<86400?Math.floor(s/3600)+'h ago':Math.floor(s/86400)+'d ago'; };
  const fmt8 = iso => new Date(iso).toLocaleString('en-GB',{day:'2-digit',month:'short',year:'numeric',hour:'2-digit',minute:'2-digit'});

  async function fetchSummary(d) { return fetch(`/analytics/summary?days=${d}`).then(r=>r.json()); }
  async function fetchBookings(d) { return fetch(`/analytics/bookings?days=${d}`).then(r=>r.json()); }
 
  function renderKPIs(d,days) {
    document.getElementById('kpi-conversations').textContent=fmt(d.conversations);
    document.getElementById('kpi-meetings').textContent=fmt(d.meetings);
    document.getElementById('kpi-cvr').textContent=pct(d.cvr);
    document.getElementById('kpi-messages').textContent=fmt(d.messages);
    const sub=`Last ${days} day${days>1?'s':''}`;
    document.getElementById('kpi-conv-sub').textContent=sub;
    document.getElementById('kpi-mtg-sub').textContent=sub;
  }
 
  function renderChart(daily) {
    const labels=daily.map(d=>new Date(d.date).toLocaleDateString('en-GB',{day:'numeric',month:'short'}));
    const values=daily.map(d=>d.conversations);
    const ctx=document.getElementById('conv-chart').getContext('2d');
    if(chart) chart.destroy();
    chart=new Chart(ctx,{type:'line',data:{labels,datasets:[{label:'Conversations',data:values,fill:true,
      backgroundColor:c=>{const g=c.chart.ctx.createLinearGradient(0,0,0,180);g.addColorStop(0,'rgba(124,58,237,0.22)');g.addColorStop(1,'rgba(124,58,237,0)');return g;},
      borderColor:'#7c3aed',borderWidth:2,pointBackgroundColor:'#7c3aed',pointRadius:3,pointHoverRadius:5,tension:0.4}]},
      options:{responsive:true,aspectRatio:4,layout:{padding:{left:0,right:0,top:0,bottom:0}},interaction:{mode:'index',intersect:false},
        plugins:{legend:{display:false},tooltip:{backgroundColor:'#13131f',borderColor:'#1f1f35',borderWidth:1,
          titleColor:'#e2e2f0',bodyColor:'#a78bfa',padding:10,
          titleFont:{family:"'Plus Jakarta Sans',sans-serif",weight:'600'},bodyFont:{family:"'Plus Jakarta Sans',sans-serif"}}},
        scales:{x:{grid:{color:'rgba(31,31,53,0.8)'},border:{display:false},ticks:{color:'#6b6b9a',font:{family:"'Plus Jakarta Sans',sans-serif",size:11},maxTicksLimit:8,padding:8}},
                y:{grid:{color:'rgba(31,31,53,0.8)'},border:{display:false},ticks:{color:'#6b6b9a',font:{family:"'Plus Jakarta Sans',sans-serif",size:11},precision:0,padding:8},beginAtZero:true}}}});
  }
 
  function renderFunnel(f) {
    const max=f.visitors||1;
    document.getElementById('f-visitors').textContent=fmt(f.visitors);
    document.getElementById('f-sent').textContent=fmt(f.sent);
    document.getElementById('f-engaged').textContent=fmt(f.engaged);
    document.getElementById('f-booked').textContent=fmt(f.booked);
    setTimeout(()=>{
      document.getElementById('fb-sent').style.width=(f.sent/max*100)+'%';
      document.getElementById('fb-engaged').style.width=(f.engaged/max*100)+'%';
      document.getElementById('fb-booked').style.width=(f.booked/max*100)+'%';
    },100);
  }
 
  function renderLatest(bookings) {
    const el=document.getElementById('latest-mtg-body');
    if(!bookings||!bookings.length){el.innerHTML='<div class="mtg-placeholder"><div class="mtg-placeholder-icon">📅</div><div class="mtg-placeholder-text">No bookings yet.</div></div>';return;}
    const b=bookings[0];
    el.innerHTML=`<div class="mtg-card">
      <div class="mtg-card__name">${esc(b.name)}</div>
      <div class="mtg-card__email">${esc(b.email)}</div>
      <div class="mtg-card__row"><span class="mtg-card__row-icon">🕐</span> Booked ${ago(b.booked_at)}</div>
      <div class="mtg-card__row"><span class="mtg-card__row-icon">📆</span> Meeting: ${fmt8(b.meeting_time)}</div>
      <div class="mtg-card__badge">✓ ${esc(b.status.charAt(0).toUpperCase()+b.status.slice(1))}</div>
    </div>`;
  }
 
  function renderTable(bookings) {
    const tbody=document.getElementById('bookings-tbody');
    document.getElementById('booking-count-tag').textContent=`${bookings.length} Total`;
    if(!bookings.length){tbody.innerHTML='<tr><td colspan="5" style="text-align:center;color:var(--c-text-muted);padding:32px 12px;">No bookings in the last 7 days.</td></tr>';return;}
    const googleCalIcon=`<svg width="14" height="14" viewBox="0 0 24 24" fill="none" xmlns="http://www.w3.org/2000/svg" style="flex-shrink:0">
      <rect x="3" y="3" width="18" height="18" rx="2" fill="#fff" stroke="#dadce0" stroke-width="1.5"/>
      <rect x="3" y="3" width="18" height="5.5" rx="2" fill="#4285F4"/>
      <rect x="3" y="6" width="18" height="2.5" fill="#4285F4"/>
      <text x="12" y="18.5" text-anchor="middle" font-family="Arial,sans-serif" font-size="8" font-weight="700" fill="#4285F4">31</text>
    </svg>`;
    tbody.innerHTML=bookings.map(b=>{
      const calUrl=`https://calendar.google.com/calendar/render?action=TEMPLATE&text=${encodeURIComponent('Discovery Meeting – '+b.name)}&dates=${toGCalDate(b.meeting_time)}/${toGCalDate(b.meeting_time,30)}&details=${encodeURIComponent('Booked via SuriBot')}&add=${encodeURIComponent(b.email)}`;
      return `<tr>
        <td><div class="td-file"><div class="td-file-icon" style="background:var(--c-purple-dim);color:var(--c-purple-l)">👤</div><div class="td-file-name">${esc(b.name)}</div></div></td>
        <td class="td-muted">${esc(b.email)}</td>
        <td class="td-muted">${fmt8(b.booked_at)}</td>
        <td class="td-muted">${fmt8(b.meeting_time)}</td>
        <td><a href="${calUrl}" target="_blank" class="btn-cal">${googleCalIcon} View in Calendar</a></td>
      </tr>`;
    }).join('');
  }
 
  function toGCalDate(iso, addMins=0) {
    const d=new Date(new Date(iso).getTime()+addMins*60000);
    return d.toISOString().replace(/[-:]/g,'').split('.')[0]+'Z';
  }
 
  async function load(days) {
    const [summary,bookings]=await Promise.all([fetchSummary(days),fetchBookings(days)]);
    renderKPIs(summary,days); renderChart(summary.daily);
    renderTable(bookings);
  }
 
  document.querySelectorAll('.date-btn').forEach(btn=>btn.addEventListener('click',()=>{
    document.querySelectorAll('.date-btn').forEach(b=>b.classList.remove('active'));
    btn.classList.add('active'); range=parseInt(btn.dataset.range,10); load(range);
  }));
 
  document.getElementById('refresh-bookings-btn').addEventListener('click',async()=>{
    const b=await fetchBookings(); renderTable(b);
  });
 
  load(range);
})();