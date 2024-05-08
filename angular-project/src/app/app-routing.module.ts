import { NgModule } from '@angular/core';
import { RouterModule, Routes } from '@angular/router';
import {AppComponent} from "./app.component";
import {DatasetAnalyzerComponent} from "./dataset-analyzer/dataset-analyzer.component";
import {RunAnalyzerComponent} from "./run-analyzer/run-analyzer.component";
import {RecommendationsComponent} from "./recommendations/recommendations.component";

const routes: Routes = [
  {
    path: '',
    component: AppComponent,
  },
   {
     path: 'dataset-analyzer',
     component: DatasetAnalyzerComponent
   },
     {
     path: 'run-analyzer',
     component: RunAnalyzerComponent
   },
    {
     path: 'recommendations',
     component: RecommendationsComponent
   }
];

@NgModule({
  imports: [RouterModule.forRoot(routes)],
  exports: [RouterModule]
})
export class AppRoutingModule { }
