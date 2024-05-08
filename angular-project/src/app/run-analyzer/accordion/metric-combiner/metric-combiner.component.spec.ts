import { ComponentFixture, TestBed } from '@angular/core/testing';

import { MetricCombinerComponent } from './metric-combiner.component';

describe('MetricCombinerComponent', () => {
  let component: MetricCombinerComponent;
  let fixture: ComponentFixture<MetricCombinerComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [MetricCombinerComponent]
    });
    fixture = TestBed.createComponent(MetricCombinerComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
