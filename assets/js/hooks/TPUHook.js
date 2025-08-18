import * as d3 from "d3";
let TPUHook = {
    updated() {
        d3.select(this.el).select("svg").remove();
        this.mounted(); // re-run animation
      },
      
      
    mounted() {
      const gridSize = 6;
      const cellSize = 50;
      const delayPerWave = 200;
  
      const svg = d3.select(this.el)
        .append("svg")
        .attr("width", gridSize * cellSize + 100)
        .attr("height", gridSize * cellSize + 100);
  
      // Draw MAC grid
      let cells = [];
      for (let row = 0; row < gridSize; row++) {
        for (let col = 0; col < gridSize; col++) {
          let cell = svg.append("rect")
            .attr("x", col * cellSize + 50)
            .attr("y", row * cellSize + 50)
            .attr("width", cellSize - 5)
            .attr("height", cellSize - 5)
            .attr("fill", "#ddd")
            .attr("stroke", "#444")
            .attr("rx", 8)
            .attr("ry", 8);
          cells.push({ row, col, element: cell });
        }
      }
  
      // Input Arrows (Left → Right)
      for (let row = 0; row < gridSize; row++) {
        svg.append("text")
          .attr("x", 10)
          .attr("y", row * cellSize + 75)
          .attr("font-size", "16px")
          .text("→")
          .transition()
          .duration(500)
          .delay(row * 100)
          .ease(d3.easeLinear)
          .attr("x", 40)
          .transition()
          .duration(0)
          .attr("x", 10)
          .on("end", function repeat() {
            d3.select(this)
              .transition()
              .duration(500)
              .attr("x", 40)
              .transition()
              .duration(0)
              .attr("x", 10)
              .on("end", repeat);
          });
      }
  
      // Input Arrows (Top → Bottom)
      for (let col = 0; col < gridSize; col++) {
        svg.append("text")
          .attr("x", col * cellSize + 75)
          .attr("y", 20)
          .attr("font-size", "16px")
          .text("↓")
          .transition()
          .duration(500)
          .delay(col * 100)
          .ease(d3.easeLinear)
          .attr("y", 45)
          .transition()
          .duration(0)
          .attr("y", 20)
          .on("end", function repeat() {
            d3.select(this)
              .transition()
              .duration(500)
              .attr("y", 45)
              .transition()
              .duration(0)
              .attr("y", 20)
              .on("end", repeat);
          });
      }
  
      // Systolic wave animation (looping)
      function animateSystolicWave(step = 0) {
        for (let c of cells) {
          if (c.row + c.col === step) {
            c.element.transition()
              .duration(200)
              .attr("fill", "#4ade80") // green active
              .transition()
              .duration(200)
              .attr("fill", "#ddd");   // reset
          }
        }
  
        let nextStep = (step + 1) % (2 * gridSize - 1);
        setTimeout(() => animateSystolicWave(nextStep), delayPerWave);
      }
  
      animateSystolicWave(); // start loop
    }
  };
  
  export default TPUHook;
  