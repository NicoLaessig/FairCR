import { ComponentFixture, TestBed } from '@angular/core/testing';

import { DatasetAccordionItemComponent } from './dataset-accordion-item.component';

describe('DatasetAccordionItemComponent', () => {
  let component: DatasetAccordionItemComponent;
  let fixture: ComponentFixture<DatasetAccordionItemComponent>;

  beforeEach(() => {
    TestBed.configureTestingModule({
      declarations: [DatasetAccordionItemComponent]
    });
    fixture = TestBed.createComponent(DatasetAccordionItemComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
