import express from 'express';
import pg from 'pg';
import crypto from 'node:crypto';
import path from 'node:path';
import { fileURLToPath } from 'node:url';

const { Pool } = pg;
const __dirname = path.dirname(fileURLToPath(import.meta.url));
const app = express();
const PORT = Number(process.env.PORT || 3000);
const DATABASE_URL = process.env.DATABASE_URL;
if (!DATABASE_URL) {
  console.error('DATABASE_URL est obligatoire pour Sûrliv V5.');
  process.exit(1);
}

const pool = new Pool({
  connectionString: DATABASE_URL,
  ssl: process.env.NODE_ENV === 'production' ? { rejectUnauthorized: false } : false,
  max: 10,
  idleTimeoutMillis: 30000
});

const now = () => new Date().toISOString();
const hash = s => crypto.createHash('sha256').update(String(s)).digest('hex');
const safeJson = value => value == null ? null : (typeof value === 'string' ? JSON.parse(value) : value);

async function query(text, params=[]) { return pool.query(text, params); }

async function initDb() {
  await query(`
    CREATE TABLE IF NOT EXISTS users (
      id TEXT PRIMARY KEY,
      role TEXT NOT NULL CHECK (role IN ('admin','marchand','livreur')),
      name TEXT NOT NULL,
      phone TEXT UNIQUE,
      password_hash TEXT,
      zone TEXT,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS livreurs (
      id TEXT PRIMARY KEY,
      user_id TEXT NOT NULL UNIQUE REFERENCES users(id) ON DELETE CASCADE,
      name TEXT NOT NULL,
      zone TEXT,
      score INTEGER DEFAULT 100,
      missions INTEGER DEFAULT 0,
      available BOOLEAN DEFAULT TRUE,
      lat DOUBLE PRECISION,
      lng DOUBLE PRECISION,
      updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS orders (
      id TEXT PRIMARY KEY,
      client TEXT NOT NULL,
      phone TEXT,
      zone TEXT,
      amount INTEGER NOT NULL,
      mode TEXT,
      status TEXT NOT NULL,
      livreur TEXT,
      time TEXT,
      eta TEXT,
      proof_json JSONB,
      problem_json JSONB,
      updated_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    CREATE TABLE IF NOT EXISTS events (
      id BIGSERIAL PRIMARY KEY,
      type TEXT NOT NULL,
      payload JSONB NOT NULL,
      created_at TIMESTAMPTZ NOT NULL DEFAULT NOW()
    );
    CREATE INDEX IF NOT EXISTS idx_orders_updated_at ON orders(updated_at DESC);
  `);

  const { rows } = await query('SELECT COUNT(*)::int AS n FROM users');
  if (rows[0].n === 0) {
    const client = await pool.connect();
    try {
      await client.query('BEGIN');
      const users = [
        ['u-admin','admin','Administrateur Sûrliv','0700000000',hash('admin123'),'Abidjan'],
        ['u-marchand','marchand','Sûrliv Boutique','0700000001',hash('marchand123'),'Cocody'],
        ['u-koffi','livreur','Koffi Yao','0700000002',hash('livreur123'),'Cocody / Yopougon'],
        ['u-ibrahim','livreur','Ibrahim Coulibaly','0700000003',hash('livreur123'),'Marcory / Abobo']
      ];
      for (const u of users) await client.query('INSERT INTO users(id,role,name,phone,password_hash,zone) VALUES($1,$2,$3,$4,$5,$6)',u);
      await client.query(`INSERT INTO livreurs(id,user_id,name,zone,score,missions,available) VALUES
        ('l-koffi','u-koffi','Koffi Yao','Cocody / Yopougon',98,214,TRUE),
        ('l-ibrahim','u-ibrahim','Ibrahim Coulibaly','Marcory / Abobo',91,132,TRUE)`);
      const orders = [
        ['SGL-2026-0341','Aïcha Konan','07 01 22 33 44','Cocody Angré',12000,'cash','livre','Koffi Yao','14:12','14:30',JSON.stringify({time:'14:32',geo:'Cocody Angré · position confirmée',signature:null,clientRating:5}),null],
        ['SGL-2026-0342','Boubacar Traoré','05 44 10 92 17','Yopougon Selmer',8500,'mobile_money','route','Koffi Yao','15:05','15:22',null,null],
        ['SGL-2026-0343','Grace Kouassi','01 22 87 65 03','Marcory Zone 4',15000,'cash','attente','Ibrahim Coulibaly','15:40','16:05',null,null],
        ['SGL-2026-0344','Serge Dago','07 88 12 40 55','Abobo Baoulé',6000,'cash','probleme','Ibrahim Coulibaly','12:50','13:10',null,JSON.stringify({reason:'Client absent',note:'Personne ne répond après 3 appels.'})],
        ['SGL-2026-0345','Fatou Bamba','01 55 20 10 08','Cocody Riviera',9500,'cash','attente','Koffi Yao','16:02','16:25',null,null]
      ];
      for (const o of orders) await client.query(`INSERT INTO orders(id,client,phone,zone,amount,mode,status,livreur,time,eta,proof_json,problem_json) VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11::jsonb,$12::jsonb)`,o);
      await client.query('COMMIT');
    } catch (e) { await client.query('ROLLBACK'); throw e; }
    finally { client.release(); }
  }
}

const clients = new Set();
async function broadcast(type, payload) {
  const data = `event: ${type}\ndata: ${JSON.stringify(payload)}\n\n`;
  for (const res of clients) res.write(data);
  await query('INSERT INTO events(type,payload) VALUES($1,$2::jsonb)', [type, JSON.stringify(payload)]);
}
function mapOrder(r) {
  return {...r, proof: safeJson(r.proof_json), problem: safeJson(r.problem_json)};
}
async function snapshot() {
  const [o,l] = await Promise.all([
    query('SELECT * FROM orders ORDER BY updated_at DESC'),
    query('SELECT * FROM livreurs ORDER BY name')
  ]);
  return {orders:o.rows.map(mapOrder), livreurs:l.rows, serverTime:now()};
}

app.use(express.json({limit:'8mb'}));
app.use(express.static(__dirname));

app.get('/api/health', async (req,res)=>res.json({ok:true,version:'5.0.0',database:'postgresql',time:now()}));
app.get('/api/bootstrap', async (req,res)=>res.json(await snapshot()));
app.get('/api/orders', async (req,res)=>res.json((await snapshot()).orders));
app.get('/api/livreurs', async (req,res)=>res.json((await snapshot()).livreurs));

app.post('/api/auth/login', async (req,res)=>{
  const {phone,password} = req.body || {};
  const {rows} = await query('SELECT id,role,name,phone,zone FROM users WHERE phone=$1 AND password_hash=$2',[phone,hash(password||'')]);
  if(!rows[0]) return res.status(401).json({error:'Identifiants invalides'});
  const token=crypto.randomBytes(24).toString('hex');
  res.json({token,user:rows[0]});
});

app.post('/api/orders', async (req,res)=>{
  const o=req.body||{};
  if(!o.id||!o.client||!Number.isFinite(Number(o.amount))) return res.status(400).json({error:'Commande invalide'});
  const stamp=now();
  await query(`INSERT INTO orders(id,client,phone,zone,amount,mode,status,livreur,time,eta,proof_json,problem_json,updated_at)
    VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11::jsonb,$12::jsonb,$13)`,
    [o.id,o.client,o.phone||'',o.zone||'Abidjan',Number(o.amount),o.mode||'cash',o.status||'attente',o.livreur||null,o.time||'',o.eta||'--',o.proof?JSON.stringify(o.proof):null,o.problem?JSON.stringify(o.problem):null,stamp]);
  const saved=mapOrder((await query('SELECT * FROM orders WHERE id=$1',[o.id])).rows[0]);
  await broadcast('order.updated',saved); res.status(201).json(saved);
});

app.patch('/api/orders/:id', async (req,res)=>{
  const oldQ=await query('SELECT * FROM orders WHERE id=$1',[req.params.id]);
  if(!oldQ.rows[0]) return res.status(404).json({error:'Commande introuvable'});
  const old=mapOrder(oldQ.rows[0]), b=req.body||{}, merged={...old,...b}, stamp=now();
  await query(`UPDATE orders SET client=$1,phone=$2,zone=$3,amount=$4,mode=$5,status=$6,livreur=$7,time=$8,eta=$9,proof_json=$10::jsonb,problem_json=$11::jsonb,updated_at=$12 WHERE id=$13`,
    [merged.client,merged.phone||'',merged.zone||'',Number(merged.amount||0),merged.mode||'cash',merged.status||'attente',merged.livreur||null,merged.time||'',merged.eta||'--',merged.proof?JSON.stringify(merged.proof):null,merged.problem?JSON.stringify(merged.problem):null,stamp,old.id]);
  const saved=mapOrder((await query('SELECT * FROM orders WHERE id=$1',[old.id])).rows[0]);
  await broadcast('order.updated',saved); res.json(saved);
});

app.patch('/api/livreurs/:id', async (req,res)=>{
  const oldQ=await query('SELECT * FROM livreurs WHERE id=$1',[req.params.id]);
  if(!oldQ.rows[0]) return res.status(404).json({error:'Livreur introuvable'});
  const old=oldQ.rows[0], b=req.body||{};
  await query('UPDATE livreurs SET available=$1,lat=$2,lng=$3,updated_at=$4 WHERE id=$5',[
    b.available===undefined?old.available:!!b.available,b.lat??old.lat,b.lng??old.lng,now(),old.id]);
  const saved=(await query('SELECT * FROM livreurs WHERE id=$1',[old.id])).rows[0];
  await broadcast('livreur.updated',saved); res.json(saved);
});

app.post('/api/sync', async (req,res)=>{
  const orders=req.body?.orders||[];
  const client=await pool.connect();
  try {
    await client.query('BEGIN');
    for(const o of orders){
      if(!o?.id) continue;
      await client.query(`INSERT INTO orders(id,client,phone,zone,amount,mode,status,livreur,time,eta,proof_json,problem_json,updated_at)
        VALUES($1,$2,$3,$4,$5,$6,$7,$8,$9,$10,$11::jsonb,$12::jsonb,$13)
        ON CONFLICT(id) DO UPDATE SET client=EXCLUDED.client,phone=EXCLUDED.phone,zone=EXCLUDED.zone,amount=EXCLUDED.amount,mode=EXCLUDED.mode,status=EXCLUDED.status,livreur=EXCLUDED.livreur,time=EXCLUDED.time,eta=EXCLUDED.eta,proof_json=EXCLUDED.proof_json,problem_json=EXCLUDED.problem_json,updated_at=EXCLUDED.updated_at`,
        [o.id,o.client||'',o.phone||'',o.zone||'Abidjan',Number(o.amount||0),o.mode||'cash',o.status||'attente',o.livreur||null,o.time||'',o.eta||'--',o.proof?JSON.stringify(o.proof):null,o.problem?JSON.stringify(o.problem):null,o.updated_at||now()]);
    }
    await client.query('COMMIT');
  } catch(e){await client.query('ROLLBACK'); throw e;} finally {client.release();}
  const snap=await snapshot(); await broadcast('snapshot.updated',snap); res.json(snap);
});

app.get('/api/events', async (req,res)=>{
  res.setHeader('Content-Type','text/event-stream'); res.setHeader('Cache-Control','no-cache'); res.setHeader('Connection','keep-alive'); res.flushHeaders?.();
  res.write(`event: ready\ndata: ${JSON.stringify({time:now()})}\n\n`); clients.add(res);
  const keep=setInterval(()=>res.write(': ping\n\n'),25000);
  req.on('close',()=>{clearInterval(keep);clients.delete(res);});
});

app.get('/api/export', async (req,res)=>{
  res.setHeader('Content-Type','application/json'); res.setHeader('Content-Disposition','attachment; filename="surliv-export.json"'); res.send(JSON.stringify(await snapshot(),null,2));
});

app.use((err,req,res,next)=>{ console.error(err); res.status(500).json({error:'Erreur serveur Sûrliv'}); });

await initDb();
app.listen(PORT,()=>console.log(`Sûrliv V5: port ${PORT}`));
