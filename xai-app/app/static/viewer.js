let globalViewer = null;
let globalData = null;

const modelColors = {
  'CNN': '#3b82f6',
  'BiLSTM': '#ec4899',
  'Transformer': '#14b8a6'
};

function render3DProtein(pdbId, residueDict) {
  const container = document.getElementById('protein-viewer');
  container.innerHTML = '<div style="color:#64748B; padding: 2rem; text-align:center;">Fetching 3D structure...</div>';
  globalData = residueDict;

  fetch(`https://files.rcsb.org/download/${pdbId}.pdb`)
    .then(res => {
      if(!res.ok) throw new Error(`PDB ID ${pdbId} not found.`);
      return res.text();
    })
    .then(pdbData => {
      container.innerHTML = ''; 
      
      setTimeout(() => {
        try {
          globalViewer = $3Dmol.createViewer(container, {backgroundColor: "#F8FAFC"});
          globalViewer.addModel(pdbData, "pdb");
          
          drawHighlights();
          
          let allResis = [];
          if (residueDict.cnn) residueDict.cnn.forEach(r => allResis.push(r.resi));
          if (residueDict.bilstm) residueDict.bilstm.forEach(r => allResis.push(r.resi));
          if (residueDict.transformer) residueDict.transformer.forEach(r => allResis.push(r.resi));
          
          if (allResis.length > 0) {
            globalViewer.zoomTo({resi: allResis});
          } else {
            globalViewer.zoomTo();
          }
          globalViewer.render();
        } catch (err) {
          container.innerHTML = `<div style="color:#EF4444; padding: 2rem; text-align:center;">3D Render Error: ${err.message}</div>`;
        }
      }, 100); 
    })
    .catch(e => {
      container.innerHTML = `<div style="color:#EF4444; padding: 2rem; text-align:center;">Network Error: ${e.message}</div>`;
    });
}

function drawHighlights() {
  if (!globalViewer || !globalData) return;
  
  globalViewer.removeAllShapes();
  globalViewer.removeAllLabels();
  
  // Darker grey cartoon for contrast
  globalViewer.setStyle({}, {cartoon: {color: '#c8ccd0', opacity: 0.9}});

  const showCnn = document.getElementById('toggle-cnn').checked;
  const showBil = document.getElementById('toggle-bilstm').checked;
  const showTrf = document.getElementById('toggle-transformer').checked;
  const showConsensus = document.getElementById('toggle-consensus').checked;

  const resiMap = {};
  
  function addResis(model, data, isActive) {
    if (!isActive || !data) return;
    data.forEach(item => {
      const r = item.resi;
      if (!resiMap[r]) resiMap[r] = [];
      resiMap[r].push(model);
    });
  }

  addResis('CNN', globalData.cnn, showCnn);
  addResis('BiLSTM', globalData.bilstm, showBil);
  addResis('Transformer', globalData.transformer, showTrf);

  Object.entries(resiMap).forEach(([resiStr, models]) => {
    const resi = parseInt(resiStr);
    
    // FIX: Try to find the CA atom first. If missing, try any atom in the residue.
    let atoms = globalViewer.selectedAtoms({resi: resi, atom: 'CA'}); 
    if (atoms.length === 0) {
        atoms = globalViewer.selectedAtoms({resi: resi});
    }
    
    if (atoms.length === 0) return; // Skip if the PDB entirely lacks this residue
    const center = atoms[0]; 

    const isConsensus = models.length >= 2;

    if (isConsensus && showConsensus) {
      globalViewer.addSphere({
        center: center, radius: 7.0, color: '#FBBF24', alpha: 0.12
      });

      models.forEach((model, i) => {
        const angle = (2 * Math.PI * i) / models.length;
        globalViewer.addSphere({
          center: {
            x: center.x + 2.5 * Math.cos(angle),
            y: center.y + 2.5 * Math.sin(angle),
            z: center.z
          },
          radius: 1.5, 
          color: modelColors[model],
          alpha: 1.0
        });
      });

      globalViewer.addLabel(`Res ${resi}: ${models.join(', ')}`, {
        position: {x: center.x, y: center.y, z: center.z + 5},
        backgroundColor: 'rgba(255, 255, 255, 0.9)', 
        fontColor: '#111', 
        fontSize: 12, 
        showBackground: true,
        borderThickness: 1,
        borderColor: '#FBBF24'
      }, null, true); 
      
    } else {
      models.forEach((model, i) => {
        let offset = {x: 0, y: 0, z: 0};
        if (models.length > 1) {
          const angle = (2 * Math.PI * i) / models.length;
          offset = {x: 2.0 * Math.cos(angle), y: 2.0 * Math.sin(angle), z: 0};
        }
        
        globalViewer.addSphere({
          center: {x: center.x + offset.x, y: center.y + offset.y, z: center.z + offset.z},
          radius: 1.5, 
          color: modelColors[model]
        });
      });
    }
  });

  globalViewer.render();
}

function resetView() {
  if (globalViewer) {
    globalViewer.zoomTo();
    globalViewer.render();
  }
}

function exportPng() {
  if (globalViewer) {
    globalViewer.pngURI(function(uri) {
      const link = document.createElement('a');
      link.download = 'protein_attribution.png';
      link.href = uri;
      link.click();
    });
  }
}
