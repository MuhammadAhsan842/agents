import test from 'node:test';
import assert from 'node:assert/strict';
import {canDelegate,missingOutputs,parseOutputs} from '../extension/delegation.js';
import {renderDashboard} from '../server/dashboard.js';
test('delegation applies only to named new output files',()=>{
 for(const action of ['write_file','create_artifact','create_dashboard'])assert.equal(canDelegate({action,target:'report.html'},{willOverwrite:false},['report.html']),true);
 assert.equal(canDelegate({action:'write_file',target:'other.md'},{willOverwrite:false},['report.html']),false);
 assert.equal(canDelegate({action:'write_file',target:'report.html'},{willOverwrite:true},['report.html']),false);
 for(const action of ['click','type','rename_file'])assert.equal(canDelegate({action,target:'report.html'},{willOverwrite:false},['report.html']),false);
});
test('completion gate identifies missing files without trusting model prose',()=>{assert.deepEqual(missingOutputs(['a.docx','b.xlsx'],[{path:'a.docx'}]),['b.xlsx']);});
test('output paths reject traversal and normalize duplicates',()=>{assert.deepEqual(parseOutputs(' reports/a.docx\nreports/a.docx\nb.xlsx'),['reports/a.docx','b.xlsx']);for(const s of ['/etc/a','../a','a/.secret','a\\b'])assert.throws(()=>parseOutputs(s));});
test('dashboard computes metrics and escapes data labels',()=>{const html=renderDashboard('name,revenue\nA,20\nB,22','<script>alert(1)</script>');assert.ok(html.includes('42'));assert.ok(html.includes('2 data rows'));assert.ok(!html.includes('<script>'));assert.ok(html.includes('&lt;script&gt;'));assert.ok(html.includes("default-src 'none'"));});
