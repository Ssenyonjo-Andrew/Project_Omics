// Isolated 3D Viewer Logic
function render3DProtein(pdbId, residueIndices) {
  const container = document.getElementById('protein-viewer');
  container.innerHTML = '<div style="color:#64748B; padding: 2rem; text-align:center;">Fetching 3D structure from RCSB PDB...</div>';
  
  fetch(`https://files.rcsb.org/download/${pdbId}.pdb`)
    .then(res => {
      if(!res.ok) throw new Error(`PDB ID ${pdbId} not found.`);
      return res.text();
    })
    .then(pdbData => {
      container.innerHTML = ''; 
      
      setTimeout(() => {
        try {
          // WHITE BACKGROUND for contrast
          let viewer = $3Dmol.createViewer(container, {backgroundColor: "white"});
          viewer.addModel(pdbData, "pdb");
          
          // Base style: Light grey cartoon with high opacity
          viewer.setStyle({}, {cartoon: {color: '#CBD5E1', opacity: 0.8}});
          
          if (residueIndices && residueIndices.length > 0) {
            const validResidues = residueIndices.filter(r => r > 0 && r < 5000);
            
            // Highlight: Bright Red Cartoon + Red Sticks so it pops out
            viewer.setStyle({resi: validResidues}, {
                cartoon: {color: '#EF4444', opacity: 1.0},
                stick: {colorscheme: 'redCarbon', radius: 0.2}
            });
          }
          
          viewer.zoomTo();
          viewer.render();
        } catch (err) {
          container.innerHTML = `<div style="color:#EF4444; padding: 2rem; text-align:center;">3D Render Error: ${err.message}</div>`;
        }
      }, 100); 
    })
    .catch(e => {
      container.innerHTML = `<div style="color:#EF4444; padding: 2rem; text-align:center;">Network Error: ${e.message}</div>`;
    });
}
