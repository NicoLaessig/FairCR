import { NgModule } from '@angular/core';
import { CommonModule } from '@angular/common';
import { DatasetAnalyzerComponent } from "./dataset-analyzer.component";
import { DataTableComponent} from "./data-table/data-table.component";
import { DatasetChartComponent} from "./dataset-chart/dataset-chart.component";
import { InputFieldComponent} from "./input-field/input-field.component";
import {FormsModule} from "@angular/forms";
import {AppRoutingModule} from "../app-routing.module";

@NgModule({
  declarations: [DatasetAnalyzerComponent, InputFieldComponent,DatasetChartComponent, DataTableComponent],
  imports: [
    CommonModule,
    FormsModule,
    AppRoutingModule,
  ],
  exports: [DatasetAnalyzerComponent],
})
export class DatasetAnalyzerModule { }
