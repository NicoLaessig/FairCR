import { Component } from '@angular/core';

@Component({
  selector: 'app-metric-combiner',
  templateUrl: './metric-combiner.component.html',
  styleUrls: ['./metric-combiner.component.scss']
})
export class MetricCombinerComponent {
  combined_metric = [{accuracy : 1}]

  addMetric(){
    this.combined_metric.push({accuracy: 1});
  }

}
