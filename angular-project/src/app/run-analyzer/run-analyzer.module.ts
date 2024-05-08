import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import {RunAnalyzerComponent} from "./run-analyzer.component";
import {RunInputFieldComponent} from "./run-input-field/run-input-field.component";
import {FormsModule} from "@angular/forms";
import { AccordionComponent } from './accordion/accordion.component';
import { DatasetAccordionItemComponent } from './accordion/dataset-accordion-item/dataset-accordion-item.component';
import { MetricScatterAccordionItemComponent } from './accordion/metric-scatter-accordion-item/metric-scatter-accordion-item.component';
import { MetricBarchartAccordionItemComponent } from './accordion/metric-barchart-accordion-item/metric-barchart-accordion-item.component';
import { MetricBoxplotAccordionItemComponent } from './accordion/metric-boxplot-accordion-item/metric-boxplot-accordion-item.component';
import {NgApexchartsModule} from "ng-apexcharts";
import { MetricCombinerComponent } from './accordion/metric-combiner/metric-combiner.component';
import { ClusterDataAccordionItemComponent } from './accordion/cluster-data-accordion-item/cluster-data-accordion-item.component';
import { RecommendationAccordionItemComponent } from './accordion/recommendation-accordion-item/recommendation-accordion-item.component';




@NgModule({
  declarations: [RunAnalyzerComponent, RunInputFieldComponent,  AccordionComponent, DatasetAccordionItemComponent, MetricScatterAccordionItemComponent, MetricBarchartAccordionItemComponent, MetricBoxplotAccordionItemComponent, MetricCombinerComponent, ClusterDataAccordionItemComponent, RecommendationAccordionItemComponent],
    imports: [
        CommonModule,
        FormsModule,
        NgApexchartsModule
    ],
  exports: [RunAnalyzerComponent],
})
export class RunAnalyzerModule { }
