import { ComponentFixture, TestBed } from '@angular/core/testing';

import { DatasetAnalyzerComponent } from './dataset-analyzer.component';

describe('DatasetAnalyzerComponent', () => {
  let component: DatasetAnalyzerComponent;
  let fixture: ComponentFixture<DatasetAnalyzerComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [DatasetAnalyzerComponent]
    });
    fixture = TestBed.createComponent(DatasetAnalyzerComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
