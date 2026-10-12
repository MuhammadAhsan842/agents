import test from 'node:test';import assert from 'node:assert/strict';import {parseCSV,summarizeCSV} from '../server/csv.js';
test('CSV quotes, commas and embedded newlines',()=>assert.deepEqual(parseCSV('name,note\r\n"A, B","hello\nworld"\r\n"C""D",ok'),[['name','note'],['A, B','hello\nworld'],['C"D','ok']]));
test('CSV metrics are computed, not generated',()=>{const r=summarizeCSV('month,revenue\nJune,20\nJuly,22');assert.equal(r.rowCount,2);assert.equal(r.columns[1].sum,42);assert.equal(r.columns[1].mean,21);assert.equal(r.columns[0].numeric,false);});
test('malformed CSV fails explicitly',()=>{assert.throws(()=>parseCSV('"broken'),/Unclosed/);assert.throws(()=>summarizeCSV('a,b\n1,2,3'),/inconsistent/);});
